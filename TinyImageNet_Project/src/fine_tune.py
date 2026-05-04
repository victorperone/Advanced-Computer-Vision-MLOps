"""
fine_tune.py
============

Two-phase transfer learning training script.

Phase 1 — HEAD ONLY (fast, high LR):
    The pretrained backbone is completely frozen.
    Only the Dense classification head learns.
    Goal: teach the head to work with ImageNet features without
          destroying them.  Duration: ~15 epochs.

Phase 2 — FINE-TUNING (slow, very low LR):
    The last N backbone layers are unfrozen.
    KEY: LR must be 10–20x lower than Phase 1 to avoid
         catastrophic forgetting.

Fixes applied vs previous version
----------------------------------
1.  get_datasets() returns 4 values — unpacked correctly.
2.  Phase 1 extended to 15 epochs (was 10) with patience=6.
3.  Phase 2 unfreezes 100 layers (was 30) — covers all
    meaningful feature extraction blocks in EfficientNetV2B0
    and ConvNeXtTiny.
4.  Phase 2 patience raised to cfg['patience'] from config.yaml.
5.  ReduceLROnPlateau patience raised to 6 (was 4) to give
    the model more room before halving the LR.
6.  Output softmax cast to float32 explicitly so mixed
    precision doesn't cause silent accuracy loss.
"""

import sys
import os
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_global_seed

cfg = load_config(profile="desktop")

if cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] CPU MODE")

import tensorflow as tf
import keras
from keras import mixed_precision

from src.models import get_model, set_backbone_trainable
from src.train import (
    get_datasets, OneCycleScheduler,
    DynamicHistoryLogger, ExperimentTracker,
)

SEED = cfg.get("seed", 42)
set_global_seed(SEED)

if cfg["use_gpu"]:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision enabled (float16 compute, float32 weights)")


def fine_tune():
    """Full two-phase transfer learning pipeline."""

    # ── 1. Datasets ──────────────────────────────────────────────────────────
    # get_datasets returns 4 values: (train_ds, val_ds, class_names, val_paths)
    # We only need train_ds and val_ds here; the others are used by evaluate_model.
    train_ds, val_ds, _class_names, _val_paths = get_datasets(cfg)

    # ── 2. Build model ────────────────────────────────────────────────────────
    kwargs_key = f"{cfg['model_type']}_kwargs"
    model_kwargs = cfg.get(kwargs_key, {})

    print(
        f"\n🏗️  Building {cfg['model_type'].upper()} "
        f"with pretrained ImageNet weights..."
    )

    model = get_model(
        model_name=cfg["model_type"],
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
        num_classes=cfg["num_classes"],
        pretrained=True,
        **model_kwargs,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Head only (backbone frozen)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📍 PHASE 1: Head-only training (backbone frozen)")
    print("=" * 60)

    set_backbone_trainable(model, trainable=False)

    # Compile with a moderate LR — the head starts from random init
    # so it can tolerate a higher LR than the backbone can in Phase 2.
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

    # 15 epochs gives the head enough time to learn stable representations
    # before we start touching the backbone.
    phase1_epochs = 15
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

    phase1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,                 # was 4 — give head more time
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
                f"phase1_{datetime.datetime.now().strftime('%H%M%S')}",
            ),
            histogram_freq=0,
        ),
    ]

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=phase1_callbacks,
    )

    phase1_best = max(history_phase1.history["val_accuracy"])
    print(f"\n✅ Phase 1 complete — best val accuracy: {phase1_best:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Partial backbone unfreeze (fine-tuning)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📍 PHASE 2: Fine-tuning (last 100 backbone layers unfrozen)")
    print("=" * 60)

    # Unfreeze the last 100 layers.
    #
    # WHY 100 instead of 30?
    #   EfficientNetV2B0 has ~270 layers total.  Only unfreezing 30 (~11%)
    #   leaves most feature extraction frozen and limits how much the model
    #   can adapt to Tiny ImageNet's domain (64×64 px, different statistics
    #   from full ImageNet).  Unfreezing 100 (~37%) covers all the later
    #   MBConv blocks that extract high-level semantic features — exactly
    #   what needs to adapt for 200-class classification.
    #   ConvNeXtTiny has fewer layers so 100 covers most of the network,
    #   which is fine because ConvNeXt's features are more robust.
    #
    # The low LR (1e-4) is what keeps catastrophic forgetting at bay,
    # not the layer count.
    set_backbone_trainable(model, trainable=True, num_layers_to_unfreeze=100)

    fine_tune_lr = 1e-4   # 10× lower than Phase 1

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=fine_tune_lr,
            weight_decay=1e-5,
            clipnorm=1.0,           # gradient clipping — critical for stability
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    save_name = f"{cfg['model_type']}_finetuned_best.keras"
    checkpoint_path = os.path.join(cfg["models_dir"], save_name)

    phase2_callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["patience"],       # from config.yaml (default 10)
            restore_best_weights=True,
            verbose=1,
        ),
        # ReduceLROnPlateau as safety net — halves LR when stuck.
        # patience=6 gives the model 6 full epochs to escape a plateau
        # before cutting the LR.  Too low and LR drops too fast.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=6,                     # was 4
            min_lr=1e-7,
            verbose=1,
        ),
        DynamicHistoryLogger(cfg=cfg),
        ExperimentTracker(cfg=cfg),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                cfg["logs_dir"], "tensorboard",
                f"phase2_{datetime.datetime.now().strftime('%H%M%S')}",
            ),
            histogram_freq=1,
        ),
    ]

    phase2_epochs = cfg["epochs"]

    print(
        f"\n🔥 Starting Phase 2 fine-tuning "
        f"for up to {phase2_epochs} epochs..."
    )
    print(f"   Learning rate : {fine_tune_lr}  (was 1e-3 in Phase 1)")
    print(f"   Patience      : {cfg['patience']} epochs")

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=phase2_callbacks,
    )

    phase2_best = max(history_phase2.history["val_accuracy"])

    print(f"\n🏆 Fine-tuning complete!")
    print(f"   Phase 1 best : {phase1_best:.4f}  ({phase1_best * 100:.1f}%)")
    print(f"   Phase 2 best : {phase2_best:.4f}  ({phase2_best * 100:.1f}%)")
    print(
        f"   Improvement  : "
        f"+{(phase2_best - phase1_best) * 100:.1f} pp"
    )
    print(f"\n💾 Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    fine_tune()
