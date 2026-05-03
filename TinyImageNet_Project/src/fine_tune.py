"""
fine_tune.py

Two-phase transfer learning training script.

Phase 1 — HEAD ONLY (fast, high LR):
  The pretrained backbone (EfficientNet/ConvNeXt) is completely frozen.
  Only the new Dense classification head learns.
  Goal: teach the head to classify Tiny ImageNet features without
        destroying the ImageNet weights in the backbone.
  Duration: ~10 epochs is usually enough.

Phase 2 — FINE-TUNING (slow, very low LR):
  The last N layers of the backbone are unfrozen.
  The entire model adapts slowly to Tiny ImageNet.
  KEY: learning rate must be 10-20x lower than Phase 1.
       Too high and you'll 'catastrophically forget' ImageNet features.
  Duration: ~20-40 epochs with early stopping.

Why does this work so well?
  ImageNet has 1.2 million images and 1000 classes.
  EfficientNet has already learned to detect edges, textures, object parts.
  We're essentially borrowing that knowledge and redirecting it to our 200 classes.
  Even though Tiny ImageNet images are only 64×64, the features transfer well.
"""

import sys
import os
import datetime
import csv
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
cfg = load_config(profile="desktop")

if cfg['use_gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] CPU MODE")

import tensorflow as tf
import keras
from keras import mixed_precision
from src.models import get_model, set_backbone_trainable
from src.train import get_datasets, OneCycleScheduler, DynamicHistoryLogger, ExperimentTracker

# Mixed precision: uses float16 on the GPU for speed, float32 for stability
# On an RTX 3060, this gives ~1.5–2x speedup with no accuracy loss
if cfg['use_gpu']:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision enabled (float16 compute, float32 weights)")


def fine_tune():
    """
    Full two-phase transfer learning pipeline.
    """
    
    # -------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------
    # get_datasets returns 4 values: datasets + metadata captured before
    # pipeline transformations strip class_names and file_paths attributes.
    train_ds, val_ds, class_names, val_file_paths = get_datasets(cfg)
    
    # -------------------------------------------------------
    # Build model (pretrained=True loads ImageNet weights)
    # -------------------------------------------------------
    kwargs_key = f"{cfg['model_type']}_kwargs"
    model_kwargs = cfg.get(kwargs_key, {})
    
    print(f"\n🏗️  Building {cfg['model_type'].upper()} with pretrained ImageNet weights...")
    
    model = get_model(
        model_name=cfg['model_type'],
        input_shape=(cfg['img_size'], cfg['img_size'], 3),
        num_classes=cfg['num_classes'],
        pretrained=True,   # Always True for fine-tuning
        **model_kwargs
    )
    
    # =========================================================
    # PHASE 1: Train classification head only
    # Backbone is frozen — only the Dense layer at the top learns
    # =========================================================
    
    print("\n" + "="*55)
    print("📍 PHASE 1: Head-only training (backbone frozen)")
    print("="*55)
    
    set_backbone_trainable(model, trainable=False)
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3,    # Higher LR is fine — head starts random
            weight_decay=1e-4,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg['label_smoothing'])
        ),
        metrics=['accuracy']
    )
    
    phase1_epochs = 10
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    
    phase1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        OneCycleScheduler(
            max_lr=cfg['max_lr'],
            total_steps=steps_per_epoch * phase1_epochs
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(cfg['logs_dir'], 'tensorboard', 
                                 f"phase1_{datetime.datetime.now().strftime('%H%M%S')}"),
            histogram_freq=0,  # Disabled in phase 1 for speed
        )
    ]
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=phase1_callbacks
    )
    
    phase1_best = max(history_phase1.history['val_accuracy'])
    print(f"\n✅ Phase 1 complete — best val accuracy: {phase1_best:.4f}")
    
    # =========================================================
    # PHASE 2: Fine-tune the last layers of the backbone
    # Very low learning rate to avoid catastrophic forgetting
    # =========================================================
    
    print("\n" + "="*55)
    print("📍 PHASE 2: Fine-tuning (last 30 backbone layers unfrozen)")
    print("="*55)
    
    # Unfreeze the last 30 layers of the backbone
    # 30 is a reasonable default — covers the final feature extraction blocks
    set_backbone_trainable(model, trainable=True, num_layers_to_unfreeze=100)
    
    # CRITICAL: Recompile with a much lower LR (10x lower than Phase 1)
    # High LR here would destroy the pretrained features we're building on
    fine_tune_lr = 1e-4   # 10x lower than phase 1
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=fine_tune_lr,
            weight_decay=1e-5,    # Lighter regularization during fine-tuning
            clipnorm=1.0          # Clip gradients to prevent spikes
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg['label_smoothing'])
        ),
        metrics=['accuracy']
    )
    
    phase2_epochs = cfg['epochs']   # Use the full epoch budget from config
    
    save_name = f"{cfg['model_type']}_finetuned_best.keras"
    checkpoint_path = os.path.join(cfg['models_dir'], save_name)
    
    phase2_callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=cfg['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        # Use a gentle cosine schedule for phase 2 — no aggressive warmup needed
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,         # Halve the LR when stuck
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        DynamicHistoryLogger(cfg=cfg),
        ExperimentTracker(cfg=cfg),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(cfg['logs_dir'], 'tensorboard',
                                 f"phase2_{datetime.datetime.now().strftime('%H%M%S')}"),
            histogram_freq=1,
        )
    ]
    
    print(f"\n🔥 Starting Phase 2 fine-tuning for up to {phase2_epochs} epochs...")
    print(f"   Learning rate: {fine_tune_lr} (was 1e-3 in Phase 1)")
    
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=phase2_callbacks
    )
    
    phase2_best = max(history_phase2.history['val_accuracy'])
    
    print(f"\n🏆 Fine-tuning complete!")
    print(f"   Phase 1 best: {phase1_best:.4f} ({phase1_best*100:.1f}%)")
    print(f"   Phase 2 best: {phase2_best:.4f} ({phase2_best*100:.1f}%)")
    print(f"   Improvement:  +{(phase2_best - phase1_best)*100:.1f} pp")
    print(f"\n💾 Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    fine_tune()