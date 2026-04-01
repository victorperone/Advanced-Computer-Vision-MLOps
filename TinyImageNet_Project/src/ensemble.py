"""
ensemble.py
===========

Ensemble inference over multiple saved models with optional
Test-Time Augmentation (TTA).

Ensembling and TTA are two of the cheapest accuracy gains available in
computer vision — they require no additional training, just more inference
passes over the validation set.

Techniques implemented
----------------------
1. **Soft voting (probability averaging)**
   Average the softmax output vectors from N models.  The class with the
   highest average probability wins.  This is almost always better than
   hard voting because it preserves the model's confidence information.

2. **Hard voting (majority vote)**
   Each model casts a vote for its top-1 class.  The class with the most
   votes wins.  Useful when models have very different calibrations.

3. **Test-Time Augmentation (TTA)**
   For each image, run several augmented versions through the model and
   average the resulting probability vectors before taking argmax.
   Augmentations used: original + horizontal flip + 4 crops.
   TTA is applied to EVERY model in the ensemble, so a 3-model ensemble
   with 6 TTA passes = 18 forward passes per image.  Slow but accurate.

Expected accuracy gains (on top of your best single model):
    Soft ensemble of 2–3 models:  +1–3 pp
    TTA alone (1 model, 6 passes): +0.5–1 pp
    TTA + ensemble:               +1.5–4 pp

Usage
-----
    # Soft vote, no TTA (fastest):
    python -m src.ensemble

    # Soft vote WITH TTA (best accuracy):
    python -m src.ensemble --tta

    # Hard vote, no TTA:
    python -m src.ensemble --method hard

PEP 8 notes
-----------
* Max line length 79 characters.
* Two blank lines between top-level definitions.
* Docstrings follow NumPy style.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config before TF to set CUDA_VISIBLE_DEVICES
from src.utils import load_config, set_global_seed  # noqa: E402

cfg = load_config(profile="laptop")  # Change to "desktop" on the big machine

if not cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras                    # noqa: E402
import tensorflow as tf         # noqa: E402

from src.train import get_datasets  # noqa: E402

set_global_seed(cfg.get("seed", 42))


# ===========================================================================
# TEST-TIME AUGMENTATION HELPERS
# ===========================================================================


def tta_augment(images):
    """
    Generate a list of augmented views of a batch for TTA.

    For each image in the batch we produce 6 views:
        0. Original (no augmentation)
        1. Horizontal flip
        2. Centre crop (90% of image, resized back)
        3. Top-left crop
        4. Top-right crop
        5. Bottom-centre crop

    The caller averages the model's softmax outputs across all 6 views.

    Why these specific augmentations?
        We want augmentations that a real photo of the object might exhibit:
        different viewpoints (crops) and reflection symmetry (flip).
        We do NOT use colour jitter or rotation here because at test time
        we want stable, reproducible predictions — only geometric transforms
        that preserve the class label with certainty.

    Parameters
    ----------
    images : tf.Tensor, shape (B, H, W, C)
        A batch of already-normalised images.

    Returns
    -------
    list[tf.Tensor]
        Six tensors of shape (B, H, W, C) — one per augmentation view.
    """
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    # View 0: original
    view0 = images

    # View 1: horizontal flip
    view1 = tf.image.flip_left_right(images)

    # Helper: crop a box from [y0, x0] with size [crop_h, crop_w] and
    # resize back to (h, w).
    def crop_and_resize(imgs, y0_frac, x0_frac, size_frac):
        """Crop a fractional region and resize back to original dimensions."""
        crop_h = tf.cast(
            tf.cast(h, tf.float32) * size_frac, tf.int32
        )
        crop_w = tf.cast(
            tf.cast(w, tf.float32) * size_frac, tf.int32
        )
        y0 = tf.cast(
            tf.cast(h - crop_h, tf.float32) * y0_frac, tf.int32
        )
        x0 = tf.cast(
            tf.cast(w - crop_w, tf.float32) * x0_frac, tf.int32
        )
        cropped = imgs[:, y0:y0 + crop_h, x0:x0 + crop_w, :]
        return tf.image.resize(cropped, [h, w])

    # View 2: centre crop (90%)
    view2 = crop_and_resize(images, 0.5, 0.5, 0.9)

    # View 3: top-left crop
    view3 = crop_and_resize(images, 0.0, 0.0, 0.85)

    # View 4: top-right crop
    view4 = crop_and_resize(images, 0.0, 1.0, 0.85)

    # View 5: bottom-centre crop
    view5 = crop_and_resize(images, 1.0, 0.5, 0.85)

    return [view0, view1, view2, view3, view4, view5]


# ===========================================================================
# PREDICTION FUNCTIONS
# ===========================================================================


def predict_with_tta(model, dataset):
    """
    Run inference over a dataset with Test-Time Augmentation.

    For each batch:
        1. Generate 6 augmented views of the batch.
        2. Run each view through the model → 6 probability matrices.
        3. Average the 6 probability matrices element-wise.
        4. The result is a single probability vector per image.

    Parameters
    ----------
    model : keras.Model
        A trained model whose output is a softmax probability vector.
    dataset : tf.data.Dataset
        Validation dataset (batched, prefetched).

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Averaged softmax probabilities for every image in the dataset.
    """
    all_probs = []

    for images, _labels in dataset:
        # Generate all augmented views of this batch
        views = tta_augment(images)

        # Accumulate softmax outputs across views
        batch_probs = np.zeros(
            (images.shape[0], cfg["num_classes"]), dtype=np.float32
        )

        for view in views:
            preds = model.predict(view, verbose=0)
            batch_probs += preds

        # Average over the number of views
        batch_probs /= len(views)
        all_probs.append(batch_probs)

    return np.concatenate(all_probs, axis=0)


def predict_without_tta(model, dataset):
    """
    Run standard inference over a dataset (single forward pass per image).

    Parameters
    ----------
    model : keras.Model
    dataset : tf.data.Dataset

    Returns
    -------
    np.ndarray, shape (N, num_classes)
        Softmax probability matrix.
    """
    all_probs = []

    for images, _labels in dataset:
        preds = model.predict(images, verbose=0)
        all_probs.append(preds)

    return np.concatenate(all_probs, axis=0)


def get_true_labels(dataset, num_classes):
    """
    Extract ground-truth labels from the validation dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Validation dataset producing (images, one_hot_labels) batches.
    num_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray, shape (N,)
        Integer class indices (argmax of one-hot vectors).
    """
    all_labels = []

    for _images, labels in dataset:
        all_labels.append(np.argmax(labels.numpy(), axis=1))

    return np.concatenate(all_labels, axis=0)


# ===========================================================================
# ENSEMBLE STRATEGIES
# ===========================================================================


def soft_vote(prob_matrices):
    """
    Average probability matrices from multiple models and return predictions.

    Soft voting is almost always better than hard voting because it uses the
    full probability distribution, not just the argmax.  When one model is
    60% confident and another is 90% confident, soft voting gives the high-
    confidence model more influence automatically.

    Parameters
    ----------
    prob_matrices : list[np.ndarray]
        One array of shape (N, num_classes) per model.

    Returns
    -------
    np.ndarray, shape (N,)
        Predicted class index for each image.
    """
    # Stack into (num_models, N, num_classes) then average across axis 0
    stacked = np.stack(prob_matrices, axis=0)
    avg_probs = stacked.mean(axis=0)
    return np.argmax(avg_probs, axis=1)


def hard_vote(prob_matrices):
    """
    Majority vote: each model votes for its top-1 class.

    If there is a tie (e.g. 2 models disagree on 2 classes), the class
    with the higher average probability among the tied classes wins
    (this is a "hard vote with soft tiebreaker").

    Parameters
    ----------
    prob_matrices : list[np.ndarray]
        One array of shape (N, num_classes) per model.

    Returns
    -------
    np.ndarray, shape (N,)
        Predicted class index for each image.
    """
    # Each model's argmax = its vote
    votes = np.stack(
        [np.argmax(p, axis=1) for p in prob_matrices], axis=0
    )   # shape: (num_models, N)

    n_images = votes.shape[1]
    final_predictions = np.zeros(n_images, dtype=np.int32)

    for i in range(n_images):
        image_votes = votes[:, i]
        # Find the mode (most common vote)
        counts = np.bincount(image_votes, minlength=cfg["num_classes"])
        max_count = counts.max()
        tied_classes = np.where(counts == max_count)[0]

        if len(tied_classes) == 1:
            final_predictions[i] = tied_classes[0]
        else:
            # Tiebreaker: pick the tied class with highest average probability
            avg_probs = np.mean(
                np.stack(prob_matrices, axis=0)[:, i, :], axis=0
            )
            best_tied = tied_classes[np.argmax(avg_probs[tied_classes])]
            final_predictions[i] = best_tied

    return final_predictions


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================


def run_ensemble(method="soft", use_tta=False):
    """
    Load all saved models, run ensemble inference, and report accuracy.

    This function automatically discovers all ``.keras`` files in the
    ``models/`` directory.  You can control which models are included by
    listing them explicitly in the ``MODEL_FILES`` list below.

    Parameters
    ----------
    method : str
        ``"soft"`` for probability averaging (recommended),
        ``"hard"`` for majority voting.
    use_tta : bool
        If True, apply Test-Time Augmentation (slower but more accurate).
    """
    # -----------------------------------------------------------------------
    # Which models to ensemble.
    # List the filenames of the models you want to combine.
    # With our new naming scheme these will look like:
    #   "vit_scratch_best.keras"
    #   "efficientnet_pretrained_best.keras"
    #   "convnext_pretrained_best.keras"
    # -----------------------------------------------------------------------
    models_dir = cfg["models_dir"]

    # Auto-discover all .keras files in the models directory
    model_files = sorted([
        f for f in os.listdir(models_dir) if f.endswith(".keras")
    ])

    if len(model_files) == 0:
        print(
            "❌ No .keras model files found in models/.\n"
            "   Train at least one model first with python -m src.train"
        )
        return

    print(f"\n🔍 Found {len(model_files)} model(s) to ensemble:")
    for name in model_files:
        print(f"   • {name}")

    if len(model_files) == 1:
        print(
            "\n⚠️  Only one model found — running single-model inference "
            f"{'with TTA' if use_tta else 'without TTA'}."
        )

    # -----------------------------------------------------------------------
    # Load the validation dataset (labels only — no training augmentation).
    # We load using cfg but we'll need the raw val_ds without MixUp/CutMix.
    # get_datasets already returns val_ds without augmentation.
    # -----------------------------------------------------------------------
    print("\n🚀 Loading validation dataset...")
    _train_ds, val_ds, _class_names, _val_paths = get_datasets(cfg)

    # Ground-truth labels (extracted before any model runs)
    print("📋 Extracting ground-truth labels...")
    y_true = get_true_labels(val_ds, cfg["num_classes"])

    # -----------------------------------------------------------------------
    # Load each model and collect its probability matrix
    # -----------------------------------------------------------------------
    prob_matrices = []
    predict_fn = predict_with_tta if use_tta else predict_without_tta

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n📦 Loading {model_file}...")

        # Custom objects needed for the ViT model's custom layers
        model = keras.models.load_model(
            model_path,
            custom_objects={
                # Register custom layers so Keras can deserialise them
                # (they are defined in models.py)
                "Patches": None,
                "PatchEncoder": None,
            },
            safe_mode=False,    # Allow loading models with Lambda layers
        )

        tta_label = "with TTA" if use_tta else "without TTA"
        print(f"   Running inference {tta_label}...")
        probs = predict_fn(model, val_ds)
        prob_matrices.append(probs)
        print(f"   Done — prediction matrix shape: {probs.shape}")

        # Free GPU/CPU memory before loading the next model
        del model
        keras.backend.clear_session()

    # -----------------------------------------------------------------------
    # Combine predictions using the chosen method
    # -----------------------------------------------------------------------
    print(f"\n🗳️  Combining predictions via {method} voting...")

    if method == "soft":
        y_pred = soft_vote(prob_matrices)
    else:
        y_pred = hard_vote(prob_matrices)

    # -----------------------------------------------------------------------
    # Compute accuracy
    # -----------------------------------------------------------------------
    accuracy = np.mean(y_pred == y_true)

    print("\n" + "=" * 50)
    print(f"  Ensemble method : {method} vote")
    print(f"  TTA             : {'yes' if use_tta else 'no'}")
    print(f"  Models combined : {len(model_files)}")
    print(f"  Val accuracy    : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("=" * 50)

    # Per-model baselines for comparison
    print("\nPer-model baselines (for reference):")
    for name, probs in zip(model_files, prob_matrices):
        single_preds = np.argmax(probs, axis=1)
        single_acc = np.mean(single_preds == y_true)
        print(f"   {name:40s}  {single_acc:.4f} ({single_acc * 100:.2f}%)")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble inference over saved models."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Voting method: 'soft' (avg probabilities) or 'hard' (majority vote).",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable Test-Time Augmentation (slower, usually more accurate).",
    )
    args = parser.parse_args()

    run_ensemble(method=args.method, use_tta=args.tta)
