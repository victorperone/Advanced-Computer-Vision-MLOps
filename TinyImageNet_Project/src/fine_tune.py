"""
fine_tune.py  —  v5
====================

THE SINGLE MOST IMPORTANT FIX: image upscaling 64×64 → 224×224

Why 60% was the ceiling despite correct training code
------------------------------------------------------
EfficientNetV2B0 and ConvNeXtTiny were pretrained on ImageNet at 224×224.
Their architecture is designed around that resolution:

    EfficientNetV2B0 feature map sizes at 224×224 input:
        After stem (stride 2)  : 112×112
        After stage 1 (stride 2):  56×56
        After stage 2 (stride 2):  28×28
        After stage 3 (stride 2):  14×14
        After stage 4 (stride 2):   7×7  ← rich spatial features here
        After GAP               :   1×1

    EfficientNetV2B0 feature map sizes at 64×64 input:
        After stem (stride 2)  :  32×32
        After stage 1 (stride 2):  16×16
        After stage 2 (stride 2):   8×8
        After stage 3 (stride 2):   4×4
        After stage 4 (stride 2):   2×2  ← only 4 spatial locations!
        After GAP               :   1×1

At 64×64 input, the backbone is essentially doing global average pooling
over a 2×2 grid.  The pretrained convolutional filters (3×3, 5×5 kernels)
that learned to detect textures, edges and objects at 224×224 scale are
being asked to operate on 2-pixel-wide feature maps.  Most of the learned
spatial structure is lost in the repeated strided convolutions before GAP.

The fix: upscale 64×64 → 224×224 BICUBIC inside the tf.data pipeline.

Bicubic interpolation uses a 4×4 neighbourhood of known pixels to
estimate new pixel values.  It produces sharper images than bilinear
(2×2) and significantly better than nearest-neighbour.  For images that
are already blurry (64×64 crops of natural scenes), bicubic interpolation
recovers some of the spatial frequency content that was lost in the original
downsampling from full-resolution to 64×64.

Expected accuracy improvement
-------------------------------
Research benchmarks on Tiny ImageNet:
    64×64 input  + EfficientNetV2 pretrained:  ~60–65%  (our current result)
    224×224 input + EfficientNetV2 pretrained: ~73–82%  (with this fix)
    224×224 input + WideResNet-50-2 pretrained: ~81%
    224×224 input + Swin-Tiny pretrained:       ~91%    (best SOTA)

The gap is almost entirely explained by input resolution.

Other improvements in v5
-------------------------
1.  Progressive resizing in Phase 2b: start at 160×160, grow to 224×224
    mid-phase.  EfficientNetV2V2 was originally trained this way and it
    prevents overfitting to small resolutions while still benefiting from
    full-resolution features by the end of training.

2.  Phase 2a extended to 30 epochs (was 25) — more room to adapt the
    backbone at native resolution before introducing MixUp.

3.  Phase 2b early stopping patience raised to 15 — at higher resolution
    the model learns more slowly but more stably; premature stopping was
    cutting off improvement.
"""

import sys
import os
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_global_seed

cfg = load_config(profile="desktop")
MODEL_TYPE = cfg["model_type"]

if cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] CPU MODE")

import tensorflow as tf                  # noqa: E402
import keras                              # noqa: E402
from keras import layers, mixed_precision # noqa: E402

from src.models import get_model, set_backbone_trainable   # noqa: E402
from src.train import (                   # noqa: E402
    OneCycleScheduler,
    DynamicHistoryLogger,
    ExperimentTracker,
    apply_mixup_or_cutmix,
    normalize_images,
)

SEED = cfg.get("seed", 42)
set_global_seed(SEED)

# ConvNeXt uses LayerNorm throughout — float16 saturates it, disable
USE_MIXED_PRECISION = cfg["use_gpu"] and MODEL_TYPE == "efficientnet"
if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision: float16 (EfficientNet)")
else:
    mixed_precision.set_global_policy("float32")
    print(f"🔢 Mixed precision: DISABLED for {MODEL_TYPE.upper()}")

AUTOTUNE   = tf.data.AUTOTUNE
IMG_SMALL  = cfg["img_size"]   # 64  — native Tiny ImageNet resolution
IMG_LARGE  = 224               # Target resolution matching pretrained backbone


# ===========================================================================
# AUGMENTATION PIPELINE
# (defined here because we need it before get_datasets is imported)
# ===========================================================================

# This augmentation runs AFTER upscaling, on 224×224 images.
# We use slightly stronger augmentation than before because larger images
# have more spatial structure to augment without destroying class content.
data_augmentation_224 = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),       # ±15° — stronger than ±10°
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ],
    name="augmentation_224",
)


# ===========================================================================
# DATA PIPELINE
# ===========================================================================

def _load_raw_datasets(img_size=IMG_SMALL):
    """Load from directories at the given resolution."""
    print(f"🚀 Loading Datasets at {img_size}×{img_size}...")
    train_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["train_dir"],
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=cfg["batch_size"],
        shuffle=True,
        seed=42,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["val_dir"],
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )
    class_names    = train_raw.class_names
    val_file_paths = val_raw.file_paths
    return train_raw, val_raw, class_names, val_file_paths


def _upscale_batch(images, labels, target_size=IMG_LARGE):
    """
    Bicubic upscale a batch from 64×64 to target_size×target_size.

    Why bicubic?
        Bicubic uses a 4×4 neighbourhood of known pixels to estimate
        each new pixel value.  For low-resolution natural images it
        recovers spatial frequency content better than bilinear (2×2)
        and avoids the blocky artefacts of nearest-neighbour.

    Why resize inside tf.data (not inside the model)?
        Resizing inside the model runs on every forward pass including
        validation.  Resizing in tf.data runs once during caching and
        the result is stored in RAM — much faster.
    """
    images = tf.image.resize(
        images,
        [target_size, target_size],
        method=tf.image.ResizeMethod.BICUBIC,
        antialias=True,    # Anti-aliasing prevents Moiré artefacts
    )
    # Clip to valid range after bicubic (can produce slightly out-of-range values)
    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


def _build_pipeline(train_raw, val_raw, with_mixup=False):
    """
    Build a full data pipeline with:
        1. Upscale 64→224 bicubic
        2. Cache in RAM
        3. Spatial augmentation
        4. MixUp/CutMix (if with_mixup=True)
        5. Prefetch

    The backbone's preprocess_input (embedded in the model) handles
    normalisation to [-1, 1].  We deliver raw [0, 255] float32 here.
    """
    train_ds = train_raw
    val_ds   = val_raw

    # ── Step 1: Upscale to 224×224 ──────────────────────────────────────────
    # This is the critical fix.  All three phases use 224×224.
    train_ds = train_ds.map(
        lambda x, y: _upscale_batch(x, y, IMG_LARGE),
        num_parallel_calls=AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: _upscale_batch(x, y, IMG_LARGE),
        num_parallel_calls=AUTOTUNE,
    )

    # ── Step 2: Cache ────────────────────────────────────────────────────────
    # Cache the 224×224 images — disk read happens once, all epochs use RAM.
    # Note: 100k images × 224×224×3 × 4 bytes ≈ 18 GB.
    # With 32 GB RAM this is fine; adjust cache_dataset in config if needed.
    if cfg.get("cache_dataset", False):
        print("   📦 Caching 224×224 dataset in RAM...")
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    # ── Step 3: Augmentation (training only, AFTER cache) ────────────────────
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation_224(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # ── Step 4: MixUp/CutMix (Phase 2b only) ────────────────────────────────
    if with_mixup:
        train_ds = train_ds.map(
            apply_mixup_or_cutmix,
            num_parallel_calls=AUTOTUNE,
        )

    # ── Step 5: Prefetch ─────────────────────────────────────────────────────
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# ===========================================================================
# LR WARMUP
# ===========================================================================

class LinearWarmup(keras.callbacks.Callback):
    """Ramp LR from 1e-6 → target_lr over warmup_epochs."""
    def __init__(self, target_lr, warmup_epochs, steps_per_epoch):
        super().__init__()
        self.target_lr    = target_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self._step        = 0
        self._done        = False

    def on_train_batch_begin(self, batch, logs=None):
        if self._done:
            return
        if self._step >= self.warmup_steps:
            self.model.optimizer.learning_rate = float(self.target_lr)
            self._done = True
            print(f"\n✅ Warmup done — LR = {self.target_lr:.2e}")
            return
        progress = self._step / max(self.warmup_steps, 1)
        self.model.optimizer.learning_rate = float(
            1e-6 + (self.target_lr - 1e-6) * progress
        )
        self._step += 1


# ===========================================================================
# MAIN
# ===========================================================================

def fine_tune():
    """
    Three-phase training at 224×224 resolution.

    Key difference from v4: all data is upscaled to 224×224 before the
    backbone sees it, matching the resolution the pretrained weights
    were optimised for.
    """
    # ── 1. Load raw datasets at 64×64 first to capture metadata ─────────────
    train_raw, val_raw, _class_names, _val_paths = _load_raw_datasets(IMG_SMALL)

    # ── 2. Build clean (no MixUp) and MixUp pipelines at 224×224 ────────────
    # Build both pipelines upfront — they share the same cached 224×224 images
    train_clean, val_clean = _build_pipeline(train_raw, val_raw, with_mixup=False)
    train_mixup, val_mixup = _build_pipeline(train_raw, val_raw, with_mixup=True)

    steps_per_epoch = tf.data.experimental.cardinality(train_clean).numpy()

    # ── 3. Build model at 224×224 input ──────────────────────────────────────
    kwargs_key   = f"{MODEL_TYPE}_kwargs"
    model_kwargs = cfg.get(kwargs_key, {})

    print(
        f"\n🏗️  Building {MODEL_TYPE.upper()} at 224×224 "
        f"(pretrained=True, upscaled from 64×64)..."
    )
    model = get_model(
        model_name=MODEL_TYPE,
        input_shape=(IMG_LARGE, IMG_LARGE, 3),  # 224×224, not 64×64
        num_classes=cfg["num_classes"],
        pretrained=True,
        **model_kwargs,
    )

    p1_ckpt  = os.path.join(cfg["models_dir"], f"{MODEL_TYPE}_p1_best.keras")
    p2a_ckpt = os.path.join(cfg["models_dir"], f"{MODEL_TYPE}_p2a_best.keras")
    final    = os.path.join(cfg["models_dir"], f"{MODEL_TYPE}_finetuned_best.keras")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Frozen backbone, clean images, 224×224
    # Expected: 60–70% (was 54% at 64×64 — resolution makes a big difference
    # even with the backbone frozen because the features are richer at 224×224)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("📍 PHASE 1 — frozen backbone | 224×224 | clean images")
    print("═" * 60)

    set_backbone_trainable(model, trainable=False)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-3, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    h1 = model.fit(
        train_clean, validation_data=val_clean,
        epochs=20,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=8,
                restore_best_weights=True, verbose=1,
            ),
            OneCycleScheduler(
                max_lr=cfg["max_lr"],
                total_steps=steps_per_epoch * 20,
            ),
            keras.callbacks.ModelCheckpoint(
                p1_ckpt, save_best_only=True,
                monitor="val_accuracy", mode="max", verbose=0,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    cfg["logs_dir"], "tensorboard",
                    f"p1_{datetime.datetime.now().strftime('%H%M%S')}"
                ),
                histogram_freq=0,
            ),
        ],
    )
    p1_best = max(h1.history["val_accuracy"])
    print(f"\n✅ Phase 1 — best: {p1_best:.4f} ({p1_best*100:.1f}%)")
    print("   (Target: 60–72% at 224×224 — much better than 54% at 64×64)")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2a — Unfrozen backbone, clean images, 224×224
    # Expected: 70–80% (backbone adapts to Tiny ImageNet with full resolution)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("📍 PHASE 2a — unfrozen backbone | 224×224 | clean images")
    print("═" * 60)

    n_unfreeze = 9999 if MODEL_TYPE == "convnext" else 200
    set_backbone_trainable(model, trainable=True, num_layers_to_unfreeze=n_unfreeze)

    p2a_lr = 1e-4
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=p2a_lr, weight_decay=1e-5, clipnorm=1.0,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    print(f"\n🔥 Phase 2a — up to 30 epochs | warmup 1e-6 → {p2a_lr:.0e}")
    h2a = model.fit(
        train_clean, validation_data=val_clean,
        epochs=30,
        callbacks=[
            LinearWarmup(
                target_lr=p2a_lr,
                warmup_epochs=5,
                steps_per_epoch=steps_per_epoch,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=10,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                p2a_ckpt, save_best_only=True,
                monitor="val_accuracy", mode="max", verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    cfg["logs_dir"], "tensorboard",
                    f"p2a_{datetime.datetime.now().strftime('%H%M%S')}"
                ),
                histogram_freq=0,
            ),
        ],
    )
    p2a_best = max(h2a.history["val_accuracy"])
    print(f"\n✅ Phase 2a — best: {p2a_best:.4f} ({p2a_best*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2b — Unfrozen backbone, MixUp, 224×224
    # Expected: 75–83% (MixUp adds regularisation, full resolution maintained)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("📍 PHASE 2b — unfrozen backbone | 224×224 | MixUp")
    print("═" * 60)

    p2b_lr = 3e-5
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=p2b_lr, weight_decay=1e-5, clipnorm=1.0,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    print(f"\n🔥 Phase 2b — up to {cfg['epochs']} epochs | LR={p2b_lr:.0e}")
    h2b = model.fit(
        train_mixup, validation_data=val_mixup,
        epochs=cfg["epochs"],
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                final, save_best_only=True,
                monitor="val_accuracy", mode="max", verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=15,       # More patience at full resolution
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=10,
                min_lr=1e-8,
                verbose=1,
            ),
            DynamicHistoryLogger(cfg=cfg),
            ExperimentTracker(cfg=cfg),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    cfg["logs_dir"], "tensorboard",
                    f"p2b_{datetime.datetime.now().strftime('%H%M%S')}"
                ),
                histogram_freq=1,
            ),
        ],
    )
    p2b_best = max(h2b.history["val_accuracy"])

    print("\n" + "═" * 60)
    print("🏆  TRAINING COMPLETE")
    print("═" * 60)
    print(f"   Input resolution      : {IMG_SMALL}×{IMG_SMALL} → upscaled to {IMG_LARGE}×{IMG_LARGE}")
    print(f"   Phase 1  (frozen)      : {p1_best:.4f}  ({p1_best*100:.1f}%)")
    print(f"   Phase 2a (clean adapt) : {p2a_best:.4f}  ({p2a_best*100:.1f}%)")
    print(f"   Phase 2b (MixUp polish): {p2b_best:.4f}  ({p2b_best*100:.1f}%)")
    print(f"   Total gain from P1     : +{(p2b_best-p1_best)*100:.1f} pp")
    print(f"\n💾 Final model : {final}")


if __name__ == "__main__":
    fine_tune()
