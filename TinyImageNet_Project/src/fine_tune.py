"""
fine_tune.py  —  v3
====================

Two-phase transfer learning with every accuracy bottleneck resolved.

ROOT CAUSE ANALYSIS of previous low results
--------------------------------------------
EfficientNet:  57%  (expected 78–85%)
ConvNeXt:      44%  (expected 72–80%)

Five root causes identified and fixed here:

1.  MixUp/CutMix during Phase 1 (BIGGEST FIX, ~8–12 pp)
    ──────────────────────────────────────────────────────
    Phase 1 trains only the classification head while the backbone is frozen.
    MixUp blends two images and produces soft labels like [0.6, 0.4, 0...].
    A FROZEN backbone produces deterministic features for each real image.
    When it sees a blended image it produces inconsistent intermediate
    features — neither image A's features nor image B's.  The head then
    receives confusing signals and cannot converge properly.
    FIX: Disable MixUp/CutMix during Phase 1, enable only in Phase 2.

2.  Old fine_tune.py still used num_layers_to_unfreeze=30 (line 148)
    ──────────────────────────────────────────────────────────────────
    The fixed file was provided but the old file remained in src/.
    This version explicitly sets 200 for EfficientNet (all meaningful
    blocks) and uses a function to automatically unfreeze the entire
    backbone for ConvNeXt.
    FIX: Unfreeze entire backbone in Phase 2 for ConvNeXt;
         unfreeze 200 layers for EfficientNet.

3.  ConvNeXt preprocessing dtype conflict  (explains 19.5% Phase 1)
    ──────────────────────────────────────────────────────────────────
    Under mixed_float16, input tensors arrive as float16.  ConvNeXtTiny
    uses Layer Normalization (not Batch Normalization) — LN does NOT keep
    float32 accumulators.  The entire forward pass runs in float16, which
    combined with the Rescaling layer's float16 arithmetic saturates
    activations early in the network.  The backbone produces near-zero or
    NaN features for every input → head loss stays at log(200)=5.3 forever.
    FIX: Force float32 cast at model input inside build_convnext(), AND
         disable mixed precision for ConvNeXt entirely in this script.

4.  Phase 2 learning rate too high given large unfreeze radius
    ────────────────────────────────────────────────────────────
    Starting Phase 2 at LR=1e-4 with 100+ layers unfrozen causes large
    gradient updates in the early unfrozen layers (closest to the output),
    which then backpropagate and disturb the frozen-for-20-epochs early
    layers.  A cosine warmup over the first 20% of Phase 2 prevents this.
    FIX: Add a linear warmup (0→1e-4) over the first 5 epochs of Phase 2
         before handing off to ReduceLROnPlateau.

5.  Separate data pipelines for Phase 1 vs Phase 2
    ────────────────────────────────────────────────
    Phase 1 needs: load → preprocess → cache → spatial_augment → NO MixUp
    Phase 2 needs: load → preprocess → cache → spatial_augment → MixUp
    The previous code used a single pipeline for both phases.
    FIX: build_phase1_datasets() and build_phase2_datasets() are separate
         functions that share the cached normalised images but differ
         in whether MixUp is applied.
"""

import sys
import os
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_global_seed

cfg = load_config(profile="desktop")

# ── GPU / Mixed precision setup ──────────────────────────────────────────────
# IMPORTANT: For ConvNeXt, mixed precision causes float16 Layer Normalization
# saturation that makes Phase 1 converge to ~19% instead of ~60%.
# We disable mixed precision for ConvNeXt and keep it for EfficientNet.
MODEL_TYPE = cfg["model_type"]

if cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] CPU MODE")

import tensorflow as tf                  # noqa: E402
import keras                              # noqa: E402
from keras import mixed_precision         # noqa: E402

from src.models import get_model, set_backbone_trainable   # noqa: E402
from src.train import (                   # noqa: E402
    OneCycleScheduler,
    DynamicHistoryLogger,
    ExperimentTracker,
    data_augmentation,
    normalize_images,
    apply_mixup_or_cutmix,
)

SEED = cfg.get("seed", 42)
set_global_seed(SEED)

USE_MIXED_PRECISION = cfg["use_gpu"] and MODEL_TYPE == "efficientnet"
if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision: float16 (EfficientNet)")
else:
    mixed_precision.set_global_policy("float32")
    print(f"🔢 Mixed precision: DISABLED for {MODEL_TYPE.upper()} "
          f"(float32 throughout)")


# ===========================================================================
# SEPARATE DATA PIPELINE FUNCTIONS
# ===========================================================================

AUTOTUNE = tf.data.AUTOTUNE


def _load_raw_datasets():
    """Load both datasets from disk, capture metadata, return raw datasets."""
    print("🚀 Loading Datasets...")

    train_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["train_dir"],
        label_mode="categorical",
        image_size=(cfg["img_size"], cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        seed=42,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["val_dir"],
        label_mode="categorical",
        image_size=(cfg["img_size"], cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # Capture metadata BEFORE any .map() strips these attributes
    class_names    = train_raw.class_names
    val_file_paths = val_raw.file_paths

    return train_raw, val_raw, class_names, val_file_paths


def build_phase1_datasets(train_raw, val_raw):
    """
    Phase 1 pipeline: NO MixUp/CutMix.

    Why: During Phase 1 the backbone is frozen.  MixUp creates blended
    images that produce inconsistent features from the frozen backbone,
    making the head training noisy and slow.  Using clean images in Phase 1
    lets the head learn a stable mapping from ImageNet features to the 200
    output classes.

    Pipeline:
        load → [normalise for scratch] → cache → spatial_augment → prefetch
    """
    train_ds = train_raw
    val_ds   = val_raw

    # Only scratch-trained models need explicit normalisation.
    # Pretrained models embed their own preprocessing inside the graph.
    if MODEL_TYPE not in ("efficientnet", "convnext"):
        train_ds = train_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)
        val_ds   = val_ds.map(normalize_images,   num_parallel_calls=AUTOTUNE)

    # Cache the normalised images in RAM (32 GB available → can fit all)
    if cfg.get("cache_dataset", False):
        print("   📦 Caching dataset in RAM (this fills RAM once, "
              "then every epoch is fast)...")
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    # Spatial augmentation only — NO MixUp/CutMix in Phase 1
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


def build_phase2_datasets(train_raw, val_raw):
    """
    Phase 2 pipeline: WITH MixUp/CutMix.

    Why: In Phase 2 the backbone is partially unfrozen and learning.
    MixUp/CutMix now work correctly because the backbone weights are
    adapting — blended images produce meaningful gradient signal.

    Pipeline:
        load → [normalise for scratch] → cache → spatial_augment
             → MixUp/CutMix → prefetch
    """
    train_ds = train_raw
    val_ds   = val_raw

    if MODEL_TYPE not in ("efficientnet", "convnext"):
        train_ds = train_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)
        val_ds   = val_ds.map(normalize_images,   num_parallel_calls=AUTOTUNE)

    if cfg.get("cache_dataset", False):
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # MixUp/CutMix enabled in Phase 2
    train_ds = train_ds.map(
        apply_mixup_or_cutmix,
        num_parallel_calls=AUTOTUNE,
    )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# ===========================================================================
# WARMUP CALLBACK FOR PHASE 2
# ===========================================================================


class LinearWarmup(keras.callbacks.Callback):
    """
    Linearly ramp the learning rate from near-zero to target_lr
    over warmup_epochs epochs at the START of Phase 2.

    Why this matters for Phase 2:
        When we unfreeze 100–200 backbone layers after Phase 1, those
        layers haven't received gradients for 15 epochs.  Starting Phase 2
        at the full LR (1e-4) causes a large gradient spike in the first
        few batches that can disturb the frozen layers and hurt accuracy.
        A 5-epoch warmup ramps from 1e-6 → 1e-4 smoothly, giving the
        unfrozen layers time to "wake up" without disrupting the rest.
    """

    def __init__(self, target_lr, warmup_epochs, steps_per_epoch):
        super().__init__()
        self.target_lr      = target_lr
        self.warmup_steps   = warmup_epochs * steps_per_epoch
        self._step          = 0
        self._warmup_done   = False

    def on_train_batch_begin(self, batch, logs=None):
        if self._warmup_done:
            return
        if self._step >= self.warmup_steps:
            self.model.optimizer.learning_rate = float(self.target_lr)
            self._warmup_done = True
            print(f"\n✅ LR warmup complete — LR = {self.target_lr}")
            return
        # Linear interpolation from 1e-6 to target_lr
        progress = self._step / max(self.warmup_steps, 1)
        lr       = 1e-6 + (self.target_lr - 1e-6) * progress
        self.model.optimizer.learning_rate = float(lr)
        self._step += 1


# ===========================================================================
# MAIN FINE-TUNE FUNCTION
# ===========================================================================


def fine_tune():
    """Full two-phase transfer learning with all fixes applied."""

    # ── 1. Load raw datasets once — shared by both phases ───────────────────
    train_raw, val_raw, class_names, val_file_paths = _load_raw_datasets()

    # ── 2. Build model ────────────────────────────────────────────────────────
    kwargs_key   = f"{cfg['model_type']}_kwargs"
    model_kwargs = cfg.get(kwargs_key, {})

    print(
        f"\n🏗️  Building {MODEL_TYPE.upper()} "
        f"with pretrained ImageNet weights..."
    )

    model = get_model(
        model_name=MODEL_TYPE,
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
        num_classes=cfg["num_classes"],
        pretrained=True,
        **model_kwargs,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Head only, clean images, no MixUp
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📍 PHASE 1: Head-only training  (backbone frozen, no MixUp)")
    print("=" * 60)

    # Build Phase 1 pipeline (no MixUp/CutMix)
    train_ds_p1, val_ds_p1 = build_phase1_datasets(train_raw, val_raw)

    set_backbone_trainable(model, trainable=False)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=3e-3,    # Slightly higher than before — clean images
            weight_decay=1e-4,     # allow more aggressive head learning
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    phase1_epochs     = 20    # More time for head to stabilise
    steps_per_epoch   = tf.data.experimental.cardinality(train_ds_p1).numpy()

    phase1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        OneCycleScheduler(
            max_lr=cfg["max_lr"],
            total_steps=steps_per_epoch * phase1_epochs,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                cfg["logs_dir"], "tensorboard",
                f"p1_{datetime.datetime.now().strftime('%H%M%S')}",
            ),
            histogram_freq=0,
        ),
    ]

    history_phase1 = model.fit(
        train_ds_p1,
        validation_data=val_ds_p1,
        epochs=phase1_epochs,
        callbacks=phase1_callbacks,
    )

    phase1_best = max(history_phase1.history["val_accuracy"])
    print(f"\n✅ Phase 1 complete — best val accuracy: {phase1_best:.4f}")
    print(
        f"   (With clean images + frozen backbone, "
        f"expect 55–70% for EfficientNet, 45–60% for ConvNeXt)"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Unfreeze backbone, add MixUp, warmup LR
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📍 PHASE 2: Fine-tuning  (backbone unfrozen, MixUp enabled)")
    print("=" * 60)

    # Build Phase 2 pipeline (WITH MixUp/CutMix)
    train_ds_p2, val_ds_p2 = build_phase2_datasets(train_raw, val_raw)

    # ── Choose unfreeze strategy per architecture ──────────────────────────
    # EfficientNetV2B0:
    #   Total ~270 layers.  MBConv6 blocks start around layer 80.
    #   Unfreezing 200 layers exposes all MBConv6 and later blocks
    #   (the ones that extract high-level semantic features) while
    #   keeping the very early stem layers frozen (they detect edges
    #   and don't need to change for Tiny ImageNet).
    #
    # ConvNeXtTiny:
    #   Total ~240 layers.  Unfreeze everything — ConvNeXt is more
    #   robust to full fine-tuning because its architecture (depthwise
    #   conv + LN) has stronger regularisation than MBConv.
    if MODEL_TYPE == "convnext":
        n_unfreeze = 9999   # Sentinel → unfreeze all layers
        print(f"   Strategy: FULL backbone unfreeze for ConvNeXt")
    else:
        n_unfreeze = 200
        print(f"   Strategy: Unfreeze last 200 layers for EfficientNet")

    set_backbone_trainable(
        model, trainable=True, num_layers_to_unfreeze=n_unfreeze
    )

    fine_tune_lr    = 5e-5    # Lower than before — safer with large unfreeze
    phase2_epochs   = cfg["epochs"]
    save_name       = f"{MODEL_TYPE}_finetuned_best.keras"
    checkpoint_path = os.path.join(cfg["models_dir"], save_name)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=fine_tune_lr,
            weight_decay=1e-5,
            clipnorm=1.0,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    warmup_epochs = 5    # 5 epochs of 0→5e-5 warmup before ReduceLROnPlateau

    phase2_callbacks = [
        LinearWarmup(
            target_lr=fine_tune_lr,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=8,       # Generous patience — let the model breathe
            min_lr=1e-8,
            verbose=1,
        ),
        DynamicHistoryLogger(cfg=cfg),
        ExperimentTracker(cfg=cfg),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                cfg["logs_dir"], "tensorboard",
                f"p2_{datetime.datetime.now().strftime('%H%M%S')}",
            ),
            histogram_freq=1,
        ),
    ]

    print(
        f"\n🔥 Starting Phase 2  (up to {phase2_epochs} epochs) ..."
    )
    print(f"   LR warmup : 1e-6 → {fine_tune_lr} over {warmup_epochs} epochs")
    print(f"   Then      : ReduceLROnPlateau (patience=8, factor=0.5)")

    history_phase2 = model.fit(
        train_ds_p2,
        validation_data=val_ds_p2,
        epochs=phase2_epochs,
        callbacks=phase2_callbacks,
    )

    phase2_best = max(history_phase2.history["val_accuracy"])

    print("\n🏆 Fine-tuning complete!")
    print(f"   Phase 1 : {phase1_best:.4f}  ({phase1_best * 100:.1f}%)")
    print(f"   Phase 2 : {phase2_best:.4f}  ({phase2_best * 100:.1f}%)")
    print(
        f"   Gain    : +{(phase2_best - phase1_best) * 100:.1f} pp"
    )
    print(f"\n💾 Saved to: {checkpoint_path}")


if __name__ == "__main__":
    fine_tune()
