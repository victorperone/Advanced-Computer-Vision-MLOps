"""
train.py
========

Main training entry-point for the Tiny ImageNet classification project.

Usage
-----
    # Laptop (CPU, 1 epoch debug run):
    python -m src.train

    # Desktop (GPU, full run — change profile in load_config below):
    python -m src.train

Workflow
--------
    1. Load hardware profile from config.yaml.
    2. Disable GPU if running on laptop.
    3. Build tf.data pipelines with augmentation.
    4. Instantiate the model from the factory in models.py.
    5. Compile with AdamW + label smoothing.
    6. Attach callbacks: GlobalCheckpoint, EarlyStopping, OneCycleScheduler,
       ExperimentTracker, DynamicHistoryLogger, TensorBoard.
    7. Fit and log.

PEP 8 notes
-----------
* Max line length 79 characters.
* Two blank lines between top-level definitions.
* Docstrings follow NumPy style.
"""

import csv
import datetime
import math
import os
import sys

# ---------------------------------------------------------------------------
# STEP 1 — Add project root to sys.path so relative imports work regardless
# of where the script is invoked from.
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# STEP 2 — Load config BEFORE importing TensorFlow.
# TensorFlow reads CUDA_VISIBLE_DEVICES at import time; if we set it after
# TF loads, the GPU selection has no effect.
# ---------------------------------------------------------------------------
from src.utils import load_config  # noqa: E402 (must stay above TF import)

# Change "laptop" → "desktop" when running on the RTX 3060.
cfg = load_config(profile="desktop")

if not cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] LAPTOP MODE — CPU only")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE — GPU enabled")

# ---------------------------------------------------------------------------
# STEP 3 — Now it is safe to import TensorFlow and Keras.
# ---------------------------------------------------------------------------
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import mixed_precision  # noqa: E402

from src.evaluation import evaluate_model  # noqa: E402
from src.models import get_model  # noqa: E402
from src.plotting import generate_training_plot  # noqa: E402
from src.utils import set_global_seed  # noqa: E402

# ---------------------------------------------------------------------------
# STEP 4 — Set global random seed for full reproducibility.
#
# Without this, results differ on every run because:
#   - Python's random module uses an OS-time seed by default.
#   - NumPy's RNG is seeded independently.
#   - TensorFlow's op-level RNG is seeded independently.
#   - The tf.data shuffle uses its own seed (we already pass seed=42
#     to image_dataset_from_directory, but the global seed covers the rest).
#
# Two runs on the same machine with the same config will now produce
# identical loss curves — a hard requirement for reproducible experiments.
#
# Note: GPU training remains non-deterministic due to cuDNN parallel
# reductions.  This seed fixes CPU runs fully; on GPU it significantly
# reduces (but cannot fully eliminate) run-to-run variance.
# ---------------------------------------------------------------------------
SEED = cfg.get("seed", 42)
set_global_seed(SEED)

# ---------------------------------------------------------------------------
# Mixed precision: on GPU, use float16 for compute and float32 for weight
# storage.  This gives ~1.5–2× throughput on Ampere/Turing cards (RTX 3060)
# with no accuracy penalty because the master weights stay float32.
# On CPU this is a no-op (TF ignores it gracefully).
# ---------------------------------------------------------------------------
if cfg["use_gpu"]:
    mixed_precision.set_global_policy("mixed_float16")
    print("⚡ Mixed precision policy: mixed_float16")


# ===========================================================================
# SECTION 1 — DATA PIPELINE HELPERS
# ===========================================================================


def normalize_images(images, labels):
    """
    Standardise pixel values using ImageNet channel mean and std.

    This is a significant upgrade over a plain /255 rescale.

    Why standardisation instead of just /255?
        Neural networks converge faster and reach higher accuracy when the
        input distribution is centred at zero with unit variance.  Simple
        /255 gives values in [0, 1], but the mean is ~0.45 — not zero.
        Standardisation shifts and scales each colour channel independently
        using the statistics measured over the entire ImageNet dataset:
            mean = [0.485, 0.456, 0.406]  (R, G, B)
            std  = [0.229, 0.224, 0.225]  (R, G, B)
        After this transform the inputs are roughly N(0, 1) per channel,
        which is exactly what the weight initialisers (Glorot, He) assume.

        In practice this is worth roughly +0.5–1 pp accuracy vs plain /255,
        and it makes training noticeably more stable in early epochs.

    Used for scratch-trained models (ResNet, ViT).
    Pretrained models (EfficientNet, ConvNeXt) embed their own preprocessing
    inside the model graph and must NOT be processed here — they would
    receive double-normalised inputs and lose several pp of accuracy.

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
    # ImageNet channel-wise statistics (standard values used across all
    # modern vision papers — ResNet, ViT, DeiT, etc.)
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # Step 1: cast to float32 and rescale to [0, 1]
    images = tf.cast(images, tf.float32) / 255.0

    # Step 2: subtract per-channel mean and divide by per-channel std
    # Broadcasting handles the (H, W, 3) shape automatically.
    images = (images - mean) / std

    return images, labels


def mixup(images, labels, alpha=0.4):
    """
    Apply MixUp data augmentation to a batch of images.

    MixUp creates a convex combination of two random training examples:
        mixed_x = λ·x_i  + (1−λ)·x_j
        mixed_y = λ·y_i  + (1−λ)·y_j
    where λ ~ Beta(alpha, alpha).

    This forces the model to learn smooth decision boundaries and reduces
    overconfidence.  It is consistently worth +1–2 pp on Tiny ImageNet.

    Why alpha=0.4 (not 0.2)?
        Higher alpha → λ is sampled closer to 0.5 more often → more mixing.
        0.4 is the sweet spot found by the DeiT paper for small datasets.
        For very clean datasets (ImageNet-scale) 0.1–0.2 is used.

    IMPORTANT — call this AFTER .cache() in the pipeline.
    If called before cache, the same mixed images are reused every epoch,
    defeating the purpose of the augmentation.

    Parameters
    ----------
    images : tf.Tensor, shape (B, H, W, C)
        Batch of images (already normalised/preprocessed).
    labels : tf.Tensor, shape (B, num_classes)
        One-hot label batch.
    alpha : float
        Beta distribution concentration parameter.  Higher → more mixing.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        Mixed (images, labels) with the same shapes as the inputs.
    """
    batch_size = tf.shape(images)[0]

    # Sample λ from Beta(alpha, alpha).
    # tf.random.uniform approximates this for the common case alpha < 1.
    # For a proper Beta sample: lam = tfp.distributions.Beta(alpha, alpha).sample()
    # but we avoid the tensorflow-probability dependency here.
    lam = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

    # Create a randomly shuffled version of the batch to mix with
    indices = tf.random.shuffle(tf.range(batch_size))

    # Cast lam to match image dtype (float16 under mixed precision)
    lam = tf.cast(lam, images.dtype)
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = tf.cast(lam, labels.dtype) * labels + (1.0 - tf.cast(lam, labels.dtype)) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def cutmix(images, labels, alpha=1.0):
    """
    Apply CutMix data augmentation to a batch of images.

    CutMix cuts a rectangular patch from one image and pastes it onto
    another, mixing the labels proportionally to the area ratio:

        mixed_x = x_i with a rectangular hole filled from x_j
        mixed_y = (1 − ratio)·y_i  +  ratio·y_j
        ratio   = patch_area / total_area

    CutMix is complementary to MixUp: MixUp blends globally, CutMix
    forces the network to classify from partial views, which strongly
    improves localisation.  Combining both (alternating randomly) is
    the approach used in the DeiT and CaiT papers.

    Parameters
    ----------
    images : tf.Tensor, shape (B, H, W, C)
    labels : tf.Tensor, shape (B, num_classes)
    alpha : float
        Beta distribution parameter controlling the size of the cut box.
        alpha=1.0 gives uniform box sizes (common default).

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
    """
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]

    # Sample λ ~ Beta(alpha, alpha) to determine the box area ratio
    lam = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

    # Box size: sqrt(1 - λ) fraction of image dimensions
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)

    # Random centre for the box
    cx = tf.random.uniform(shape=[], minval=0, maxval=img_w, dtype=tf.int32)
    cy = tf.random.uniform(shape=[], minval=0, maxval=img_h, dtype=tf.int32)

    # Clamp box to image boundaries
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

    # Build a binary mask: 0 inside the box, 1 outside
    # We construct it via padding so we stay in pure TF ops (no loops).
    inner_h = y2 - y1
    inner_w = x2 - x1

    # Zeros inside the cut rectangle
    inner = tf.zeros([batch_size, inner_h, inner_w, tf.shape(images)[3]])

    # Pad to full image size
    paddings = tf.stack([
        [0, 0],
        [y1, img_h - y2],
        [x1, img_w - x2],
        [0, 0],
    ])
    mask = tf.pad(inner, paddings, constant_values=1.0)   # 1 = keep original

    # Shuffle the batch to get the "donor" images
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)

    # Blend: original where mask=1, donor where mask=0
    # Cast mask to image dtype so multiply works under mixed precision
    mask = tf.cast(mask, images.dtype)
    mixed_images = images * mask + shuffled_images * (1.0 - mask)

    # Recalculate λ from actual box area (may differ from sampled λ due to clamping)
    actual_ratio = tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(
        img_h * img_w, tf.float32
    )
    mixed_labels = (
        (1.0 - actual_ratio) * labels
        + actual_ratio * tf.gather(labels, indices)
    )

    return mixed_images, mixed_labels


def apply_mixup_or_cutmix(images, labels, mixup_prob=0.5):
    """
    Randomly apply either MixUp or CutMix to a batch.

    Using both augmentations together (50/50 split) gives better results
    than either alone because they regularise different aspects:
        MixUp  → smoother global decision boundaries.
        CutMix → forces classification from partial views (better localisation).

    Parameters
    ----------
    images : tf.Tensor
    labels : tf.Tensor
    mixup_prob : float
        Probability of choosing MixUp; complement probability uses CutMix.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
    """
    use_mixup = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < mixup_prob
    images, labels = tf.cond(
        use_mixup,
        lambda: mixup(images, labels),
        lambda: cutmix(images, labels),
    )
    return images, labels


# ---------------------------------------------------------------------------
# Data augmentation pipeline — applied only to the training set.
# Each transformation is explained below.
# ---------------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        # Flip left-right: the most reliable augmentation for image datasets.
        # Doubles effective training examples at zero cost.
        keras.layers.RandomFlip("horizontal"),

        # Small rotation: ±10° (0.1 radians).  Objects don't always sit
        # perfectly upright in photos.
        keras.layers.RandomRotation(0.1),

        # Zoom in/out up to ±10%.  Teaches scale invariance.
        keras.layers.RandomZoom(0.1),

        # Shift the image up/down and left/right by up to 10%.
        # Teaches position invariance; works well with 64×64 images.
        keras.layers.RandomTranslation(0.1, 0.1),

        # Randomly change brightness by ±20%.
        # Makes the model invariant to lighting conditions.
        keras.layers.RandomBrightness(0.2),

        # Randomly change saturation and hue (colour jitter).
        # Improves robustness to camera white-balance differences.
        # factor=0.15 is mild enough not to corrupt class identity.
        keras.layers.RandomContrast(0.15),
    ],
    name="data_augmentation_pipeline",
)


def get_datasets(cfg):
    """
    Build train and validation ``tf.data.Dataset`` objects.

    Pipeline order (order matters for correctness and speed):
        1. Load from directory  — decode JPEG, resize to img_size × img_size.
        2. Capture metadata     — save class_names and file_paths NOW, before
                                  any .map() call strips these attributes.
        3. Normalise            — ImageNet standardisation for scratch models;
                                  skip for pretrained (they self-preprocess).
        4. Cache                — store in RAM (fast epoch 2+).
        5. Augment              — AFTER cache so transforms re-randomise each epoch.
        6. MixUp / CutMix       — AFTER cache for the same reason.
        7. Prefetch             — overlap CPU data loading with model compute.

    Why capture metadata before transformations?
        ``image_dataset_from_directory`` returns a ``DirectoryIteratorDataset``
        which has ``.class_names`` and ``.file_paths`` attributes.  As soon
        as you call ``.map()``, ``.cache()``, or ``.prefetch()``, the result
        is a new dataset type (``MapDataset``, ``CacheDataset``, etc.) that
        does NOT carry those attributes forward.  Accessing ``.class_names``
        on a ``_PrefetchDataset`` raises AttributeError — the bug you hit.
        The fix is to read both values immediately after loading, before any
        pipeline transformation.

    Why cache before augmentation?
        If we augmented before caching, the cached dataset would contain
        one fixed augmented version of each image.  The network would see
        the same augmented image every epoch — no benefit.  Caching the raw
        (normalised) images and augmenting on-the-fly gives a freshly
        augmented view each epoch.

    Parameters
    ----------
    cfg : dict
        Active hardware profile loaded by ``load_config()``.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, list[str], list[str]]
        (train_ds, val_ds, class_names, val_file_paths)

        ``class_names``   — sorted list of 200 class folder names.
        ``val_file_paths``— flat list of every validation image path, in the
                            same order as the validation dataset.  Needed by
                            ``evaluate_model`` in evaluation.py.
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

    # -----------------------------------------------------------------------
    # IMPORTANT: Capture metadata NOW — before any pipeline transformation.
    # After .map() / .cache() / .prefetch() these attributes no longer exist.
    # -----------------------------------------------------------------------
    class_names = train_ds_raw.class_names          # e.g. ['n01443537', ...]
    val_file_paths = val_ds_raw.file_paths           # flat list of image paths

    AUTOTUNE = tf.data.AUTOTUNE

    # Start the pipeline from the raw datasets
    train_ds = train_ds_raw
    val_ds = val_ds_raw

    # --- Normalisation (scratch-trained models only) ---
    # Pretrained EfficientNet and ConvNeXt embed their own preprocessing
    # inside the model graph (see models.py).  Applying standardisation on
    # top of that would corrupt their expected input range.
    if cfg["model_type"] not in ("efficientnet", "convnext"):
        train_ds = train_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)

    # --- Cache after normalisation ---
    # Stores the entire normalised dataset in RAM so we never re-read disk.
    # Only enable this when you have enough free RAM (see config.yaml).
    if cfg.get("cache_dataset", False):
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # --- Spatial / colour augmentation (training only, AFTER cache) ---
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # --- MixUp + CutMix (training only, AFTER cache) ---
    # These operate on full batches so they must come after batching, which
    # image_dataset_from_directory already handles.
    train_ds = train_ds.map(
        apply_mixup_or_cutmix,
        num_parallel_calls=AUTOTUNE,
    )

    # --- Prefetch: prepare the next batch while the model processes the current ---
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, val_file_paths


# ===========================================================================
# SECTION 2 — CUSTOM TRAINING CALLBACKS
# ===========================================================================


class OneCycleScheduler(keras.callbacks.Callback):
    """
    Implement the 1-Cycle learning rate policy (Leslie Smith, 2018).

    The schedule has two phases:
        Phase A (first 30% of training): LR rises linearly from ~0 to max_lr.
        Phase B (remaining 70%):         LR decays via cosine annealing back to ~0.

    Why does this work?
        The warmup phase prevents large early gradient updates that can send
        weights to bad regions of the loss surface.
        The cosine decay provides a smooth final descent into a minimum.
        Together they give faster convergence and higher final accuracy than
        a constant LR or simple decay.

    Parameters
    ----------
    max_lr : float
        Peak learning rate (the ``max_lr`` key from config.yaml).
    total_steps : int
        Total number of gradient update steps over the whole training run
        (steps_per_epoch × max_epochs).
    """

    def __init__(self, max_lr, total_steps):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self._step = 0

    def on_train_batch_begin(self, batch, logs=None):
        """Update the optimiser's learning rate before each gradient step."""
        # Progress: 0.0 at start → 1.0 at the end of training
        pct = min(self._step / max(self.total_steps, 1), 1.0)

        if pct < 0.3:
            # --- Phase A: linear warm-up ---
            lr = self.max_lr * (pct / 0.3)
        else:
            # --- Phase B: cosine annealing ---
            progress = (pct - 0.3) / 0.7
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.max_lr * cosine_factor

        # Clamp to a safe minimum to avoid numerical instability
        lr = max(lr, 1e-7)
        self.model.optimizer.learning_rate = float(lr)
        self._step += 1


class ExperimentTracker(keras.callbacks.Callback):
    """
    Append one row of experiment metadata to ``logs/experiment_logs.csv``
    at the end of each training run.

    Tracked columns: profile, date, time window, duration, model name,
    pretrained flag, epochs run, best train/val loss and accuracy, dropout,
    learning rate, batch size, weight decay, label smoothing, history file.

    Parameters
    ----------
    cfg : dict
        Active hardware profile.
    log_filename : str
        Name of the CSV file inside the logs directory.
    """

    def __init__(self, cfg, log_filename="experiment_logs.csv"):
        super().__init__()
        self.cfg = cfg
        self.log_file = os.path.join(cfg["logs_dir"], log_filename)
        self.best_val_acc = 0.0
        self.best_metrics = {}

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
        end_time = datetime.datetime.now()
        duration = end_time - self._start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        # --- Read dropout from the correct kwargs sub-dict ---
        kwargs_key = f"{self.cfg['model_type']}_kwargs"
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

        fieldnames = list(row.keys())
        file_exists = os.path.isfile(self.log_file)

        with open(self.log_file, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(
            "\n📊 MLOps: Experiment saved to "
            f"{os.path.basename(self.log_file)}"
        )


class DynamicHistoryLogger(keras.callbacks.Callback):
    """
    Save per-epoch training metrics to a timestamped CSV and auto-plot them.

    The CSV filename encodes the model name, date, and best validation
    accuracy so experiment artefacts are self-describing:
        vit_history_30-06-2025_0.6812.csv

    Parameters
    ----------
    cfg : dict
        Active hardware profile.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {"epoch": epoch + 1}
        for key, value in logs.items():
            row[key] = float(value)
        self._epoch_data.append(row)

    def on_train_end(self, logs=None):
        if not self._epoch_data:
            return

        best_val_acc = max(
            row.get("val_accuracy", 0.0) for row in self._epoch_data
        )

        date_str = datetime.datetime.now().strftime("%d-%m-%Y")
        model_name = self.cfg["model_type"].lower()
        filename = f"{model_name}_history_{date_str}_{best_val_acc:.4f}.csv"
        filepath = os.path.join(self.cfg["logs_dir"], filename)

        # Store filename so ExperimentTracker can reference it
        self.cfg["history_file"] = filename

        keys = list(self._epoch_data[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self._epoch_data:
                writer.writerow(row)

        print(f"📈 MLOps: Epoch history saved to logs/{filename}")

        # Auto-generate accuracy/loss plots
        generate_training_plot(
            history_path=filepath,
            model_name=model_name,
            logs_dir=self.cfg["logs_dir"],
        )


class GlobalCheckpoint(keras.callbacks.ModelCheckpoint):
    """
    ModelCheckpoint that persists the all-time best across separate runs.

    Standard ``ModelCheckpoint`` only remembers the best score from the
    current run.  If you restart training, it may overwrite a previously
    better model.  This subclass reads ``experiment_logs.csv`` at train-start
    and primes ``self.best`` with the historical high score so the file is
    only overwritten when a true new record is achieved.

    Parameters
    ----------
    filepath : str
        Path to save the model file.
    cfg : dict
        Active hardware profile (used to find experiment_logs.csv).
    **kwargs
        Forwarded to ``ModelCheckpoint.__init__``.
    """

    def __init__(self, filepath, cfg, **kwargs):
        super().__init__(filepath, **kwargs)
        self._cfg = cfg

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        log_file = os.path.join(self._cfg["logs_dir"], "experiment_logs.csv")
        historical_best = 0.0

        if os.path.isfile(log_file):
            with open(log_file, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Model", "").lower() == self._cfg["model_type"]:
                        try:
                            acc = float(row.get("Val_Accuracy", 0.0))
                            if acc > historical_best:
                                historical_best = acc
                        except ValueError:
                            pass

        # Inject the historical best into Keras's internal best tracker
        self.best = historical_best
        print(
            f"\n🏆 All-time best for {self._cfg['model_type'].upper()}: "
            f"{historical_best:.4f}"
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
    """
    Build datasets, compile model, attach callbacks, and start training.

    All hyperparameters come from the active profile in ``config.yaml``.
    To switch profiles edit the load_config call near the top of this file:
        cfg = load_config(profile="desktop")   # or "desktop"
    """
    # --- 1. Datasets ---
    # get_datasets now returns 4 values: the two pipeline datasets PLUS
    # the class_names and val_file_paths captured before transformations
    # stripped those attributes from the dataset objects.
    train_ds, val_ds, class_names, val_file_paths = get_datasets(cfg)

    # --- 2. Model ---
    # Read the architecture-specific kwargs from the YAML profile.
    # E.g. for ViT this grabs the entire vit_kwargs dict and unpacks it.
    kwargs_key = f"{cfg['model_type']}_kwargs"
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

    # --- 3. Compile ---
    # AdamW decouples the weight decay from the adaptive learning rate
    # update, which is the mathematically correct formulation (vs L2 in Adam).
    # clipnorm=1.0 prevents exploding gradients, especially useful for ViT.
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

    # --- 4. Callbacks ---
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    total_steps = steps_per_epoch * cfg["epochs"]

    # -----------------------------------------------------------------------
    # Model filename encodes BOTH the architecture AND the training regime.
    #
    # Why this matters:
    #   The old name  `vit_best.keras`  is the same whether you trained ViT
    #   from scratch or loaded ImageNet weights.  A pretrained run would
    #   silently overwrite your scratch-trained model (or vice-versa).
    #
    # Industry convention: encode every meaningful axis in the filename so
    # artefacts are self-describing and never collide.
    #   vit_scratch_best.keras        — ViT trained from random init
    #   efficientnet_pretrained_best.keras — EfficientNet + ImageNet weights
    # -----------------------------------------------------------------------
    regime = "pretrained" if cfg.get("pretrained", False) else "scratch"
    save_name = f"{cfg['model_type']}_{regime}_best.keras"
    checkpoint_path = os.path.join(cfg["models_dir"], save_name)
    print(f"💾 Checkpoint will be saved to: models/{save_name}")

    callbacks = [
        # Save the best model (beats historical record) to disk
        GlobalCheckpoint(
            filepath=checkpoint_path,
            cfg=cfg,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=0,
        ),

        # Stop early if val_accuracy stops improving; restore the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),

        # 1-Cycle LR schedule: warmup then cosine decay
        OneCycleScheduler(
            max_lr=cfg["max_lr"],
            total_steps=total_steps,
        ),

        # Append a row to experiment_logs.csv at run end
        ExperimentTracker(cfg=cfg),

        # Save per-epoch CSV + plot
        DynamicHistoryLogger(cfg=cfg),

        # TensorBoard: run `tensorboard --logdir logs/tensorboard` to view
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                cfg["logs_dir"],
                "tensorboard",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            ),
            histogram_freq=1,
        ),
    ]

    # --- 5. Train ---
    print(f"\n🔥 Starting training for up to {cfg['epochs']} epoch(s)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
    )

    # --- 6. Post-training evaluation ---
    # Use the class_names and val_file_paths captured in step 1 — these are
    # no longer available from the transformed dataset objects themselves.
    print("\n🧪 Running post-training evaluation pipeline...")
    evaluate_model(
        model=model,
        dataset=val_ds,
        class_names=class_names,
        model_name=cfg["model_type"],
        logs_dir=cfg["logs_dir"],
        image_paths=val_file_paths,
    )


# ---------------------------------------------------------------------------
# Entry-point guard — allows the module to also be imported by fine_tune.py
# without immediately starting training.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
