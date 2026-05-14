"""
fine_tune.py  —  v4
====================

Three-phase transfer learning pipeline with cache isolation fix.

ROOT CAUSE ANALYSIS of 58.75% ceiling (was expected 78–85%)
------------------------------------------------------------

Problem 1 — Dataset cache exhaustion between phases  (CRITICAL BUG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config.yaml sets cache_dataset: true.
Phase 1 calls build_phase1_datasets(train_raw, val_raw) which caches
train_raw into RAM.  After Phase 1 finishes, train_raw is an exhausted
iterator — it has been fully consumed by the cache fill.

Phase 2 then calls build_phase2_datasets(train_raw, val_raw) on the
SAME exhausted train_raw object.  tf.data silently tries to iterate
the exhausted iterator, gets no data, and trains the model on empty
batches for an entire epoch.  The model loss spikes (model trains on
nothing), weights are destroyed, and val_accuracy drops from 54% → 32%
at the start of Phase 2.

This explains the mysterious 22 pp drop at Phase 2 epoch 1 that
warmup could not fix — no amount of LR tuning helps if the model is
training on empty batches.

FIX: Reload raw datasets fresh for each phase.  Each phase calls
_load_raw_datasets() independently so it gets a fresh unconsumed
iterator.  Cache fills happen inside each phase's own pipeline.

Problem 2 — MixUp enabled immediately when backbone unfreezes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When Phase 2 starts, the unfrozen layers have received zero gradients
for 20 epochs (they were frozen during Phase 1).  Enabling MixUp
immediately on top of this means the model must simultaneously:
  (a) Wake up 200 frozen layers from zero gradient state
  (b) Learn from blended (mixed) images

Both tasks at once produce noisy, conflicting gradient signals.
The model destabilises for 10–20 epochs before settling, wasting
training budget and failing to reach the accuracy plateau.

FIX: Three-phase protocol:
  Phase 1: Backbone frozen, NO MixUp.  Head learns ImageNet features.
  Phase 2: Backbone unfrozen (last 200 layers), NO MixUp.  Backbone
           stabilises on clean images.  10–15 epochs.
  Phase 3: Backbone unfrozen, MixUp enabled.  Full fine-tuning with
           regularisation.  Until early stopping.

Problem 3 — ReduceLROnPlateau cutting LR too fast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With patience=8 the LR halved 3 times in 40 epochs, decaying to
6.25e-6 before the model had found its final optimum.  At that LR
the gradient steps are so tiny the model cannot escape local minima.

FIX: Remove ReduceLROnPlateau from Phase 3.  Use a cosine annealing
schedule from the target LR down to a small floor instead.  This
gives smooth, predictable decay without premature collapse.

Problem 4 — Phase 1 LR too high with clean images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 1 used LR=3e-3 with OneCycleScheduler peaking at max_lr=0.005.
The OneCycleScheduler peak is computed over steps_per_epoch * 20, so
the model sees a very high LR spike in epochs 4–6 that can overshoot
the optimal head weights, requiring the model to recover in later epochs.

FIX: Phase 1 uses a gentler OneCycleScheduler with max_lr=1e-3.
     This is enough to train the head quickly without overshooting.
"""

import sys
import os
import datetime
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_global_seed

cfg = load_config(profile="desktop")

# ---------------------------------------------------------------------------
# GPU / Mixed precision — must be set BEFORE importing TensorFlow.
# ---------------------------------------------------------------------------
MODEL_TYPE = cfg["model_type"]

if cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] CPU MODE")

import tensorflow as tf           # noqa: E402
import keras                       # noqa: E402
from keras import mixed_precision  # noqa: E402

from src.models import get_model, set_backbone_trainable  # noqa: E402
from src.train import (            # noqa: E402
    OneCycleScheduler,
    DynamicHistoryLogger,
    ExperimentTracker,
    data_augmentation,
    normalize_images,
    apply_mixup_or_cutmix,
)

SEED = cfg.get("seed", 42)
set_global_seed(SEED)

# ConvNeXt uses LayerNorm throughout — float16 LN saturates activations.
# EfficientNet uses BatchNorm — more robust to float16.
# We disable mixed precision for ConvNeXt to avoid the saturation bug.
USE_MIXED_PRECISION = cfg["use_gpu"] and MODEL_TYPE == "efficientnet"
if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision: float16 (EfficientNet)")
else:
    mixed_precision.set_global_policy("float32")
    print(
        f"🔢 Mixed precision: DISABLED for {MODEL_TYPE.upper()} "
        "(float32 throughout — prevents LN saturation)"
    )

AUTOTUNE = tf.data.AUTOTUNE


# ===========================================================================
# SECTION 1 — DATA PIPELINE
# ===========================================================================


def load_datasets_fresh(with_mixup: bool):
    """
    Load train and val datasets from disk and build a complete pipeline.

    This function always loads fresh from disk — it never reuses a
    previously consumed iterator.  This is critical because tf.data
    caches exhaust the source iterator; calling .cache() a second time
    on the same iterator silently returns empty batches.

    Each call to this function creates an entirely new dataset graph
    so Phase 1, Phase 2, and Phase 3 each get a fresh, correct pipeline.

    Parameters
    ----------
    with_mixup : bool
        True  → include MixUp/CutMix after spatial augmentation.
        False → spatial augmentation only, no batch blending.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, list[str], list[str]]
        (train_ds, val_ds, class_names, val_file_paths)
    """
    print(
        f"\n   📂 Loading datasets from disk "
        f"({'with' if with_mixup else 'without'} MixUp)..."
    )

    # --- Load raw batched datasets from directory structure ------------------
    train_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["train_dir"],
        label_mode="categorical",
        image_size=(cfg["img_size"], cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        seed=SEED,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["val_dir"],
        label_mode="categorical",
        image_size=(cfg["img_size"], cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # Capture metadata NOW — before any .map()/.cache() strips these attrs.
    class_names    = train_raw.class_names
    val_file_paths = val_raw.file_paths

    train_ds = train_raw
    val_ds   = val_raw

    # --- Normalisation (scratch-trained models only) -------------------------
    # EfficientNet and ConvNeXt embed their own preprocessing inside the
    # model graph (Lambda cast + preprocess_input / Rescaling), so they
    # must receive raw [0, 255] float32 pixels.  DO NOT normalise them here.
    # ResNet and ViT use our normalize_images() which applies ImageNet stats.
    if MODEL_TYPE not in ("efficientnet", "convnext"):
        train_ds = train_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)
        val_ds   = val_ds.map(normalize_images,   num_parallel_calls=AUTOTUNE)

    # --- Cache in RAM --------------------------------------------------------
    # Caching stores the normalised (or raw, for pretrained) images in RAM so
    # disk I/O only happens once.  After the first epoch, all data is served
    # from RAM.  With 32 GB DDR5 this is safe and gives ~3× epoch speedup.
    #
    # IMPORTANT: We call .cache() on this fresh dataset object, NOT on a
    # previously consumed one.  The cache will fill on the first epoch and
    # persist for the remainder of this phase.
    if cfg.get("cache_dataset", False):
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    # --- Spatial augmentation (training set only) ----------------------------
    # Applied AFTER cache so each epoch sees freshly randomised transforms.
    # If applied before cache, the same augmented images repeat every epoch.
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # --- MixUp / CutMix (optional, training set only) -----------------------
    # Applied AFTER cache and spatial augmentation for the same reason.
    # Only enabled in Phase 3 when the backbone is already stabilised.
    if with_mixup:
        train_ds = train_ds.map(
            apply_mixup_or_cutmix,
            num_parallel_calls=AUTOTUNE,
        )

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, val_file_paths


# ===========================================================================
# SECTION 2 — LEARNING RATE CALLBACKS
# ===========================================================================


class CosineDecayScheduler(keras.callbacks.Callback):
    """
    Cosine annealing learning rate schedule for Phase 3.

    Smoothly decays the learning rate from ``initial_lr`` to ``min_lr``
    following a cosine curve over ``total_steps`` gradient updates.

    Why cosine annealing instead of ReduceLROnPlateau for Phase 3?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ReduceLROnPlateau reduces the LR reactively when the metric stops
    improving — it is a "panic response" to a plateau.  The problem is
    it can halve the LR 3–4 times in quick succession, driving LR to
    sub-1e-6 values where gradient steps are too tiny for the model to
    escape local minima.

    Cosine annealing decays proactively and smoothly, giving the model
    a predictable trajectory.  The cosine shape naturally slows decay
    near the end of training, allowing fine-grained exploration near
    the final optimum without completely stopping learning.

    This is the standard LR schedule used in DeiT, EfficientNet papers,
    and most modern fine-tuning work.

    Parameters
    ----------
    initial_lr : float
        Starting learning rate (the peak after any warmup).
    min_lr : float
        Floor learning rate (never goes below this).
    total_steps : int
        Total number of gradient update steps for this phase.
        Computed as steps_per_epoch × n_epochs.
    """

    def __init__(self, initial_lr: float, min_lr: float, total_steps: int):
        super().__init__()
        self.initial_lr  = initial_lr
        self.min_lr      = min_lr
        self.total_steps = total_steps
        self._step       = 0

    def on_train_batch_begin(self, batch, logs=None):
        """Update LR before each gradient step using cosine formula."""
        progress = min(self._step / max(self.total_steps, 1), 1.0)

        # Cosine annealing: starts at 1.0, ends at 0.0
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Interpolate between initial_lr and min_lr
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        self.model.optimizer.learning_rate = float(lr)
        self._step += 1


class LinearWarmup(keras.callbacks.Callback):
    """
    Linearly ramp learning rate from near-zero to ``target_lr``.

    Used at the start of Phase 2 and Phase 3 when backbone layers that
    have not received gradients for many epochs are suddenly unfrozen.
    Starting at full LR causes a gradient spike that disturbs the
    already-learned weights.  The warmup gives the network time to
    "wake up" the unfrozen layers gently.

    Parameters
    ----------
    target_lr : float
        Final learning rate after warmup completes.
    warmup_epochs : int
        Number of epochs over which to ramp up.
    steps_per_epoch : int
        Number of gradient steps per epoch.
    """

    def __init__(
        self,
        target_lr: float,
        warmup_epochs: int,
        steps_per_epoch: int,
    ):
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
            print(f"\n   ✅ LR warmup complete — LR = {self.target_lr:.2e}")
            return

        progress = self._step / max(self.warmup_steps, 1)
        lr       = 1e-7 + (self.target_lr - 1e-7) * progress
        self.model.optimizer.learning_rate = float(lr)
        self._step += 1


# ===========================================================================
# SECTION 3 — THREE-PHASE TRAINING
# ===========================================================================


def fine_tune():
    """
    Three-phase transfer learning with isolated dataset loads per phase.

    Phase 1  →  Backbone FROZEN,   NO MixUp  (head learns ImageNet features)
    Phase 2  →  Backbone UNFROZEN, NO MixUp  (backbone stabilises)
    Phase 3  →  Backbone UNFROZEN, MixUp ON  (full fine-tune + regularisation)
    """

    # =========================================================================
    # PHASE 1 — Train classification head only
    #
    # Why: The new Dense classification head starts with random weights.
    # The pretrained backbone has carefully learned ImageNet features.
    # If we let the head's large random gradients backpropagate into the
    # backbone in epoch 1, they will corrupt the learned features.
    # Freezing the backbone means Phase 1 gradients only update the head,
    # giving it 20 epochs to learn a stable class mapping before we
    # touch the backbone.
    #
    # Why no MixUp in Phase 1:
    # The frozen backbone produces the same fixed feature vectors for each
    # image regardless of training iteration.  MixUp blends two images and
    # expects the model to produce a soft output.  But a frozen backbone
    # produces inconsistent feature vectors for blended images (it was never
    # trained on blended inputs), giving the head noisy, contradictory signals
    # and slowing convergence significantly.
    # =========================================================================
    print("\n" + "=" * 65)
    print("📍 PHASE 1 — Head-only training")
    print("   Backbone: FROZEN  |  MixUp: OFF  |  LR: OneCycle 1e-3")
    print("=" * 65)

    # Load fresh datasets for Phase 1 — NO MixUp
    train_p1, val_p1, class_names, val_file_paths = load_datasets_fresh(
        with_mixup=False
    )

    # Build the model with pretrained ImageNet weights
    kwargs_key   = f"{MODEL_TYPE}_kwargs"
    model_kwargs = cfg.get(kwargs_key, {})

    print(f"\n🏗️  Building {MODEL_TYPE.upper()} with pretrained weights...")
    model = get_model(
        model_name=MODEL_TYPE,
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
        num_classes=cfg["num_classes"],
        pretrained=True,
        **model_kwargs,
    )

    # Freeze the entire backbone — only the Dense head will train
    set_backbone_trainable(model, trainable=False)

    steps_per_epoch = tf.data.experimental.cardinality(train_p1).numpy()
    phase1_epochs   = 20

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    history_p1 = model.fit(
        train_p1,
        validation_data=val_p1,
        epochs=phase1_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            OneCycleScheduler(
                max_lr=1e-3,    # Moderate peak — head doesn't need high LR
                total_steps=steps_per_epoch * phase1_epochs,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    cfg["logs_dir"], "tensorboard",
                    f"p1_{datetime.datetime.now().strftime('%H%M%S')}",
                ),
                histogram_freq=0,
            ),
        ],
    )

    phase1_best = max(history_p1.history["val_accuracy"])
    print(f"\n✅ Phase 1 complete | Best val accuracy: {phase1_best:.4f}")

    # =========================================================================
    # PHASE 2 — Unfreeze backbone, stabilise on clean images
    #
    # Why a separate "stabilisation" phase before enabling MixUp:
    # After Phase 1, the unfrozen backbone layers have received zero gradients
    # for 20 epochs.  Their weights are exactly where ImageNet training left
    # them.  Introducing MixUp simultaneously with unfreezing means the model
    # must adapt both (a) the newly active backbone weights AND (b) learn to
    # handle blended inputs — two hard tasks at once.
    #
    # Phase 2 unfreezes the backbone but keeps clean images.  This gives the
    # backbone 10–15 epochs to adapt its features to Tiny ImageNet's domain
    # (64×64 images, different class distribution than ImageNet-1k) before
    # we add the additional regularisation pressure of MixUp.
    #
    # This "stabilise then regularise" pattern is used in Google's fine-tuning
    # recipes for production models.
    # =========================================================================
    print("\n" + "=" * 65)
    print("📍 PHASE 2 — Backbone stabilisation")
    print("   Backbone: UNFROZEN  |  MixUp: OFF  |  LR: warmup → cosine")
    print("=" * 65)

    # Load fresh datasets for Phase 2 — NO MixUp, fresh iterator
    train_p2, val_p2, _, _ = load_datasets_fresh(with_mixup=False)

    # Choose unfreeze depth per architecture
    if MODEL_TYPE == "convnext":
        # ConvNeXtTiny: unfreeze everything.  Its depthwise-conv + LN design
        # has strong implicit regularisation, making full fine-tuning safe.
        n_unfreeze = 9999
        print("   Strategy: FULL backbone unfreeze (ConvNeXt)")
    else:
        # EfficientNetV2B0: unfreeze last 200 of 270 layers.
        # This covers all MBConv6 blocks (the semantic feature extractors)
        # while keeping the very early edge-detection stem layers frozen.
        n_unfreeze = 200
        print("   Strategy: Last 200 layers unfrozen (EfficientNet)")

    set_backbone_trainable(model, trainable=True, num_layers_to_unfreeze=n_unfreeze)

    phase2_lr     = 2e-5      # Lower than Phase 1 — backbone is delicate
    phase2_epochs = 15        # Fixed: enough to stabilise without overfitting

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=phase2_lr,
            weight_decay=1e-5,
            clipnorm=1.0,     # Clip gradients — essential with large unfreeze
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    save_path = os.path.join(cfg["models_dir"], f"{MODEL_TYPE}_finetuned_best.keras")

    history_p2 = model.fit(
        train_p2,
        validation_data=val_p2,
        epochs=phase2_epochs,
        callbacks=[
            LinearWarmup(
                target_lr=phase2_lr,
                warmup_epochs=3,              # 3-epoch warmup for backbone wake-up
                steps_per_epoch=steps_per_epoch,
            ),
            CosineDecayScheduler(
                initial_lr=phase2_lr,
                min_lr=1e-7,
                total_steps=steps_per_epoch * phase2_epochs,
            ),
            keras.callbacks.ModelCheckpoint(
                save_path,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    cfg["logs_dir"], "tensorboard",
                    f"p2_{datetime.datetime.now().strftime('%H%M%S')}",
                ),
                histogram_freq=0,
            ),
        ],
    )

    phase2_best = max(history_p2.history["val_accuracy"])
    print(f"\n✅ Phase 2 complete | Best val accuracy: {phase2_best:.4f}")

    # Restore best Phase 2 weights before entering Phase 3
    print(f"   Loading best Phase 2 weights from {save_path}...")
    model.load_weights(save_path)

    # =========================================================================
    # PHASE 3 — Full fine-tuning with MixUp enabled
    #
    # At this point:
    #   - Head has learned stable class mapping (Phase 1)
    #   - Backbone has adapted to Tiny ImageNet domain (Phase 2)
    #   - Both are now ready for the additional regularisation of MixUp
    #
    # MixUp + CutMix provide strong regularisation that prevents overfitting
    # on the training set and improves generalisation on the validation set.
    # They are most effective when the model is already partially converged —
    # which is why we introduce them here rather than from the start.
    #
    # LR schedule: cosine decay from 1e-5 to 1e-8.
    # No ReduceLROnPlateau — it decays LR too aggressively and too reactively.
    # Cosine gives a smooth, predictable trajectory.
    # =========================================================================
    print("\n" + "=" * 65)
    print("📍 PHASE 3 — Full fine-tuning with MixUp")
    print("   Backbone: UNFROZEN  |  MixUp: ON  |  LR: cosine 1e-5 → 1e-8")
    print("=" * 65)

    # Load fresh datasets for Phase 3 — WITH MixUp, fresh iterator
    train_p3, val_p3, _, _ = load_datasets_fresh(with_mixup=True)

    phase3_lr     = 1e-5      # Lower than Phase 2 — final fine adjustment
    phase3_epochs = cfg["epochs"]   # Use full epoch budget; ES will stop early

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=phase3_lr,
            weight_decay=1e-5,
            clipnorm=1.0,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    history_p3 = model.fit(
        train_p3,
        validation_data=val_p3,
        epochs=phase3_epochs,
        callbacks=[
            CosineDecayScheduler(
                initial_lr=phase3_lr,
                min_lr=1e-8,
                total_steps=steps_per_epoch * phase3_epochs,
            ),
            keras.callbacks.ModelCheckpoint(
                save_path,
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
            DynamicHistoryLogger(cfg=cfg),
            ExperimentTracker(cfg=cfg),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    cfg["logs_dir"], "tensorboard",
                    f"p3_{datetime.datetime.now().strftime('%H%M%S')}",
                ),
                histogram_freq=1,
            ),
        ],
    )

    phase3_best = max(history_p3.history["val_accuracy"])

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "🏆 " * 20)
    print("Fine-tuning complete!")
    print(f"   Phase 1 (head only)       : {phase1_best:.4f}  ({phase1_best * 100:.1f}%)")
    print(f"   Phase 2 (backbone, clean) : {phase2_best:.4f}  ({phase2_best * 100:.1f}%)")
    print(f"   Phase 3 (full + MixUp)    : {phase3_best:.4f}  ({phase3_best * 100:.1f}%)")
    print(f"   Total gain P1 → P3        : +{(phase3_best - phase1_best) * 100:.1f} pp")
    print(f"\n💾 Best model saved to: {save_path}")


if __name__ == "__main__":
    fine_tune()
