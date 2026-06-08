"""
train.py
========

Main training entry-point for the Tiny ImageNet classification project.

Fixes applied vs previous version
----------------------------------
1.  MixUp: lam is cast to image dtype before multiply — fixes float16
    type mismatch under mixed precision.
2.  CutMix: mask is cast to image dtype before multiply — same fix.
3.  profile now defaults to "desktop" for the desktop machine.
4.  normalize_images casts output to float32 explicitly so ViT and
    ResNet always receive float32 inputs regardless of mixed precision
    policy (mixed precision should only affect compute inside layers,
    not the input tensor).
"""

import csv
import datetime
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config  # noqa: E402

# ── Change "laptop" → "desktop" when running on the RTX 3060 ────────────────
cfg = load_config(profile="desktop")

if not cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] LAPTOP MODE — CPU only")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import mixed_precision  # noqa: E402

from src.evaluation import evaluate_model  # noqa: E402
from src.models import get_model  # noqa: E402
from src.plotting import generate_training_plot  # noqa: E402
from src.utils import set_global_seed  # noqa: E402

SEED = cfg.get("seed", 42)
set_global_seed(SEED)

if cfg["use_gpu"]:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision policy: mixed_float16")


# ===========================================================================
# SECTION 1 — DATA PIPELINE HELPERS
# ===========================================================================


def normalize_images(images, labels):
    """
    Standardise pixel values using ImageNet channel mean and std.

    Always outputs float32 regardless of the global mixed precision policy.
    Mixed precision should affect compute inside model layers, not the
    input tensor itself — feeding float16 inputs to a scratch-trained model
    causes subtle numerical instability in early training.

    Parameters
    ----------
    images : tf.Tensor
        Batch of images, dtype uint8 or float32, range [0, 255].
    labels : tf.Tensor
        Corresponding one-hot label batch; passed through unchanged.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        (images_standardised_float32, labels).
    """
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # Always cast to float32 first — never float16 for the raw input
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std

    return images, labels


def mixup(images, labels, alpha=0.4):
    """
    Apply MixUp augmentation.

    FIX: lam is cast to images.dtype before multiply.
    Under mixed precision, images is float16 but lam is float32 (Python
    scalar). TF refuses to mix dtypes in element-wise ops, so we cast lam
    to match images.  Labels stay float32 (TF handles that automatically).
    """
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

    indices = tf.random.shuffle(tf.range(batch_size))

    # ── FIX: cast lam to match image dtype ──────────────────────────────────
    lam_img = tf.cast(lam, images.dtype)
    mixed_images = (
        lam_img * images
        + (tf.ones_like(lam_img) - lam_img) * tf.gather(images, indices)
    )

    # Labels always stay float32 for loss computation correctness
    lam_f32 = tf.cast(lam, tf.float32)
    mixed_labels = (
        lam_f32 * labels
        + (1.0 - lam_f32) * tf.gather(labels, indices)
    )

    return mixed_images, mixed_labels


def cutmix(images, labels, alpha=1.0):
    """
    Apply CutMix augmentation.

    FIX: mask is cast to images.dtype before multiply.
    Same mixed-precision dtype issue as MixUp — the mask is built as
    float32 from tf.pad but images may be float16.
    """
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    lam = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)

    cx = tf.random.uniform(shape=[], minval=0, maxval=img_w, dtype=tf.int32)
    cy = tf.random.uniform(shape=[], minval=0, maxval=img_h, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

    inner_h = y2 - y1
    inner_w = x2 - x1

    inner = tf.zeros(
        [batch_size, inner_h, inner_w, tf.shape(images)[3]],
        dtype=tf.float32,   # build in float32 first, cast below
    )
    paddings = tf.stack([
        [0, 0], [y1, img_h - y2], [x1, img_w - x2], [0, 0],
    ])
    mask_f32 = tf.pad(inner, paddings, constant_values=1.0)

    # ── FIX: cast mask to match image dtype ─────────────────────────────────
    mask = tf.cast(mask_f32, images.dtype)

    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)

    mixed_images = images * mask + shuffled_images * (tf.ones_like(mask) - mask)

    # Labels always float32
    actual_ratio = tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(
        img_h * img_w, tf.float32
    )
    mixed_labels = (
        (1.0 - actual_ratio) * labels
        + actual_ratio * tf.gather(labels, indices)
    )

    return mixed_images, mixed_labels


def apply_mixup_or_cutmix(images, labels, mixup_prob=0.5):
    """Randomly apply MixUp or CutMix (50/50 split)."""
    use_mixup = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < mixup_prob
    images, labels = tf.cond(
        use_mixup,
        lambda: mixup(images, labels),
        lambda: cutmix(images, labels),
    )
    return images, labels


# ---------------------------------------------------------------------------
# Augmentation pipeline — applied AFTER cache, BEFORE MixUp/CutMix.
# ---------------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.15),
    ],
    name="data_augmentation_pipeline",
)


def get_datasets(cfg):
    """
    Build train and validation tf.data.Dataset objects.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, list[str], list[str]]
        (train_ds, val_ds, class_names, val_file_paths)
    """
    print("🚀 Loading Datasets...")

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["train_dir"],
        label_mode="categorical",
        image_size=(cfg["img_size"], cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        seed=42,
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        cfg["val_dir"],
        label_mode="categorical",
        image_size=(cfg["img_size"], cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # Capture metadata BEFORE any .map()/.cache()/.prefetch() strips it
    class_names    = train_ds_raw.class_names
    val_file_paths = val_ds_raw.file_paths

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds_raw
    val_ds   = val_ds_raw

    # Normalise scratch-trained models only.
    # Pretrained models embed their own preprocessing inside the model graph.
    if cfg["model_type"] not in ("efficientnet", "convnext"):
        train_ds = train_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)
        val_ds   = val_ds.map(normalize_images,   num_parallel_calls=AUTOTUNE)

    if cfg.get("cache_dataset", False):
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    # Augmentation AFTER cache — re-randomised every epoch
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # MixUp/CutMix AFTER cache — same reason
    train_ds = train_ds.map(
        apply_mixup_or_cutmix,
        num_parallel_calls=AUTOTUNE,
    )

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, val_file_paths


# ===========================================================================
# SECTION 2 — CUSTOM TRAINING CALLBACKS
# ===========================================================================


class OneCycleScheduler(keras.callbacks.Callback):
    """1-Cycle LR: 30% linear warmup then 70% cosine decay."""

    def __init__(self, max_lr, total_steps):
        super().__init__()
        self.max_lr      = max_lr
        self.total_steps = total_steps
        self._step       = 0

    def on_train_batch_begin(self, batch, logs=None):
        pct = min(self._step / max(self.total_steps, 1), 1.0)
        if pct < 0.3:
            lr = self.max_lr * (pct / 0.3)
        else:
            progress      = (pct - 0.3) / 0.7
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr            = self.max_lr * cosine_factor
        lr = max(lr, 1e-7)
        self.model.optimizer.learning_rate = float(lr)
        self._step += 1


class ExperimentTracker(keras.callbacks.Callback):
    """Append one CSV row per run to logs/experiment_logs.csv."""

    def __init__(self, cfg, log_filename="experiment_logs.csv"):
        super().__init__()
        self.cfg      = cfg
        self.log_file = os.path.join(cfg["logs_dir"], log_filename)
        self.best_val_acc  = 0.0
        self.best_metrics  = {}

    def on_train_begin(self, logs=None):
        self._start_time = datetime.datetime.now()
        self._epochs_run = 0

    def on_epoch_end(self, epoch, logs=None):
        self._epochs_run += 1
        current_val_acc = logs.get("val_accuracy", 0.0)
        if current_val_acc >= self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_metrics = logs.copy()

    def on_train_end(self, logs=None):
        end_time  = datetime.datetime.now()
        duration  = end_time - self._start_time
        hours, r  = divmod(duration.seconds, 3600)
        minutes   = r // 60

        kwargs_key   = f"{self.cfg['model_type']}_kwargs"
        model_kwargs = self.cfg.get(kwargs_key, {})
        actual_dropout = model_kwargs.get(
            "dropout_rate", self.cfg.get("dropout_rate", 0.0)
        )

        row = {
            "Profile":        self.cfg.get("profile_name", "unknown").upper(),
            "Date":           self._start_time.strftime("%d/%m/%Y"),
            "Start_Time":     self._start_time.strftime("%H:%M"),
            "End_Time":       end_time.strftime("%H:%M"),
            "Duration":       f"{hours}h {minutes}m",
            "Model":          self.cfg["model_type"].upper(),
            "Pretrained":     self.cfg.get("pretrained", False),
            "Epochs_Run":     self._epochs_run,
            "Train_Loss":     round(self.best_metrics.get("loss", 0.0), 4),
            "Train_Accuracy": round(self.best_metrics.get("accuracy", 0.0), 4),
            "Val_Loss":       round(self.best_metrics.get("val_loss", 0.0), 4),
            "Val_Accuracy":   round(self.best_metrics.get("val_accuracy", 0.0), 4),
            "Dropout_Rate":   actual_dropout,
            "Learning_Rate":  self.cfg.get("learning_rate", 0.001),
            "Batch_Size":     self.cfg["batch_size"],
            "Weight_Decay":   self.cfg.get("weight_decay", 1e-4),
            "Label_Smoothing": self.cfg.get("label_smoothing", 0.1),
            "History_File":   self.cfg.get("history_file", "N/A"),
        }

        fieldnames  = list(row.keys())
        file_exists = os.path.isfile(self.log_file)

        with open(self.log_file, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"\n📊 MLOps: Experiment saved to "
            f"{os.path.basename(self.log_file)}"
        )


class DynamicHistoryLogger(keras.callbacks.Callback):
    """Save per-epoch CSV and auto-generate plots."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg         = cfg
        self._epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row  = {"epoch": epoch + 1}
        for key, value in logs.items():
            row[key] = float(value)
        self._epoch_data.append(row)

    def on_train_end(self, logs=None):
        if not self._epoch_data:
            return

        best_val_acc = max(
            row.get("val_accuracy", 0.0) for row in self._epoch_data
        )
        date_str   = datetime.datetime.now().strftime("%d-%m-%Y")
        model_name = self.cfg["model_type"].lower()
        filename   = (
            f"{model_name}_history_{date_str}_{best_val_acc:.4f}.csv"
        )
        filepath   = os.path.join(self.cfg["logs_dir"], filename)

        self.cfg["history_file"] = filename

        keys = list(self._epoch_data[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self._epoch_data:
                writer.writerow(row)

        print(f"📈 MLOps: Epoch history saved to logs/{filename}")

        generate_training_plot(
            history_path=filepath,
            model_name=model_name,
            logs_dir=self.cfg["logs_dir"],
        )


class GlobalCheckpoint(keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint that persists the all-time best across runs."""

    def __init__(self, filepath, cfg, **kwargs):
        super().__init__(filepath, **kwargs)
        self._cfg = cfg

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        log_file       = os.path.join(self._cfg["logs_dir"], "experiment_logs.csv")
        historical_best = 0.0

        if os.path.isfile(log_file):
            with open(log_file, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (
                        row.get("Model", "").lower()
                        == self._cfg["model_type"]
                    ):
                        try:
                            acc = float(row.get("Val_Accuracy", 0.0))
                            if acc > historical_best:
                                historical_best = acc
                        except ValueError:
                            pass

        self.best = historical_best
        print(
            f"\n🏆 All-time best for "
            f"{self._cfg['model_type'].upper()}: {historical_best:.4f}"
        )
        print("🛡️  New weights saved only if they beat this score.\n")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None:
            if current <= self.best:
                print(
                    f"\n🛑 Not saving — "
                    f"current {current:.4f} ≤ best {self.best:.4f}"
                )
            else:
                print(
                    f"\n🌟 NEW HIGH SCORE: {current:.4f} "
                    f"(was {self.best:.4f}) — saving model."
                )
        super().on_epoch_end(epoch, logs)


# ===========================================================================
# SECTION 3 — MAIN TRAINING FUNCTION
# ===========================================================================


def train():
    """Build datasets, compile, attach callbacks, train."""

    train_ds, val_ds, class_names, val_file_paths = get_datasets(cfg)

    kwargs_key   = f"{cfg['model_type']}_kwargs"
    model_kwargs = cfg.get(kwargs_key, {})

    print(
        f"\n🧠 Building model: {cfg['model_type'].upper()} "
        f"(pretrained={cfg.get('pretrained', False)})"
    )
    print(f"📐 Architecture blueprint: {model_kwargs}")

    model = get_model(
        model_name=cfg["model_type"],
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", False),
        **model_kwargs,
    )

    print("⚙️  Compiling with AdamW + label smoothing + gradient clipping...")
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=cfg["learning_rate"],
            weight_decay=float(cfg["weight_decay"]),
            clipnorm=1.0,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg["label_smoothing"])
        ),
        metrics=["accuracy"],
    )

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    total_steps     = steps_per_epoch * cfg["epochs"]

    regime          = "pretrained" if cfg.get("pretrained", False) else "scratch"
    save_name       = f"{cfg['model_type']}_{regime}_best.keras"
    checkpoint_path = os.path.join(cfg["models_dir"], save_name)
    print(f"💾 Checkpoint will be saved to: models/{save_name}")

    callbacks = [
        GlobalCheckpoint(
            filepath=checkpoint_path,
            cfg=cfg,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        OneCycleScheduler(
            max_lr=cfg["max_lr"],
            total_steps=total_steps,
        ),
        ExperimentTracker(cfg=cfg),
        DynamicHistoryLogger(cfg=cfg),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                cfg["logs_dir"],
                "tensorboard",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            ),
            histogram_freq=1,
        ),
    ]

    print(f"\n🔥 Starting training for up to {cfg['epochs']} epoch(s)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
    )

    print("\n🧪 Running post-training evaluation pipeline...")
    evaluate_model(
        model=model,
        dataset=val_ds,
        class_names=class_names,
        model_name=cfg["model_type"],
        logs_dir=cfg["logs_dir"],
        image_paths=val_file_paths,
    )


if __name__ == "__main__":
    train()
