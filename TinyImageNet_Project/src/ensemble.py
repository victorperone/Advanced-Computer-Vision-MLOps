"""
ensemble.py
===========

Ensemble inference with soft/hard voting and Test-Time Augmentation.

Fix applied vs previous version
---------------------------------
The load_model call previously passed:
    custom_objects={"Patches": None, "PatchEncoder": None}

Passing None as the class value causes Keras to try to call None as a
constructor when deserialising the saved model config — hence the
TypeError: 'str' object is not callable.

Fix: import the actual Patches and PatchEncoder classes from models.py
and pass them to custom_objects.  For non-ViT models these are unused
but harmless.
"""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_global_seed  # noqa: E402

cfg = load_config(profile="desktop")

if not cfg["use_gpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras                    # noqa: E402
import tensorflow as tf         # noqa: E402

# ── Import actual custom layer classes for model deserialisation ─────────────
# Keras saves model architecture as a JSON config.  When loading, it needs
# to reconstruct any custom layer classes by calling their constructors.
# Passing None instead of the class means Keras tries to call None() which
# raises TypeError: 'str' object is not callable.
from src.models import Patches, PatchEncoder  # noqa: E402
from src.train import get_datasets             # noqa: E402

set_global_seed(cfg.get("seed", 42))


# ===========================================================================
# TEST-TIME AUGMENTATION
# ===========================================================================


def tta_augment(images):
    """
    Generate 6 augmented views of a batch for TTA.

    Views: original, horizontal flip, centre crop (90%),
           top-left crop (85%), top-right crop (85%),
           bottom-centre crop (85%).
    """
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    view0 = images
    view1 = tf.image.flip_left_right(images)

    def crop_and_resize(imgs, y0_frac, x0_frac, size_frac):
        crop_h = tf.cast(tf.cast(h, tf.float32) * size_frac, tf.int32)
        crop_w = tf.cast(tf.cast(w, tf.float32) * size_frac, tf.int32)
        y0 = tf.cast(tf.cast(h - crop_h, tf.float32) * y0_frac, tf.int32)
        x0 = tf.cast(tf.cast(w - crop_w, tf.float32) * x0_frac, tf.int32)
        cropped = imgs[:, y0:y0 + crop_h, x0:x0 + crop_w, :]
        return tf.image.resize(cropped, [h, w])

    view2 = crop_and_resize(images, 0.5, 0.5, 0.9)
    view3 = crop_and_resize(images, 0.0, 0.0, 0.85)
    view4 = crop_and_resize(images, 0.0, 1.0, 0.85)
    view5 = crop_and_resize(images, 1.0, 0.5, 0.85)

    return [view0, view1, view2, view3, view4, view5]


# ===========================================================================
# PREDICTION FUNCTIONS
# ===========================================================================


def predict_with_tta(model, dataset):
    """Run inference with Test-Time Augmentation (6 views per image)."""
    all_probs = []
    for images, _labels in dataset:
        views       = tta_augment(images)
        batch_probs = np.zeros(
            (images.shape[0], cfg["num_classes"]), dtype=np.float32
        )
        for view in views:
            preds = model.predict(view, verbose=0)
            batch_probs += preds.astype(np.float32)
        batch_probs /= len(views)
        all_probs.append(batch_probs)
    return np.concatenate(all_probs, axis=0)


def predict_without_tta(model, dataset):
    """Standard single-pass inference."""
    all_probs = []
    for images, _labels in dataset:
        preds = model.predict(images, verbose=0)
        all_probs.append(preds.astype(np.float32))
    return np.concatenate(all_probs, axis=0)


def get_true_labels(dataset, num_classes):
    """Extract integer ground-truth labels from a one-hot dataset."""
    all_labels = []
    for _images, labels in dataset:
        all_labels.append(np.argmax(labels.numpy(), axis=1))
    return np.concatenate(all_labels, axis=0)


# ===========================================================================
# ENSEMBLE STRATEGIES
# ===========================================================================


def soft_vote(prob_matrices):
    """Average probability vectors then argmax."""
    stacked   = np.stack(prob_matrices, axis=0)
    avg_probs = stacked.mean(axis=0)
    return np.argmax(avg_probs, axis=1)


def hard_vote(prob_matrices):
    """Majority vote with soft tiebreaker."""
    votes    = np.stack(
        [np.argmax(p, axis=1) for p in prob_matrices], axis=0
    )
    n_images = votes.shape[1]
    final    = np.zeros(n_images, dtype=np.int32)

    for i in range(n_images):
        image_votes = votes[:, i]
        counts      = np.bincount(image_votes, minlength=cfg["num_classes"])
        max_count   = counts.max()
        tied        = np.where(counts == max_count)[0]

        if len(tied) == 1:
            final[i] = tied[0]
        else:
            avg_probs = np.mean(
                np.stack(prob_matrices, axis=0)[:, i, :], axis=0
            )
            final[i] = tied[np.argmax(avg_probs[tied])]

    return final


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================


def run_ensemble(method="soft", use_tta=False):
    """
    Load all saved models, run ensemble inference, report accuracy.

    Parameters
    ----------
    method : str
        "soft" (average probabilities) or "hard" (majority vote).
    use_tta : bool
        If True, apply Test-Time Augmentation.
    """
    models_dir  = cfg["models_dir"]
    model_files = sorted([
        f for f in os.listdir(models_dir) if f.endswith(".keras")
    ])

    if not model_files:
        print(
            "❌ No .keras model files found in models/.\n"
            "   Train at least one model first."
        )
        return

    print(f"\n🔍 Found {len(model_files)} model(s) to ensemble:")
    for name in model_files:
        print(f"   • {name}")

    if len(model_files) == 1:
        print(
            f"\n⚠️  Only one model found — single-model inference "
            f"{'with TTA' if use_tta else 'without TTA'}."
        )

    print("\n🚀 Loading validation dataset...")
    _train_ds, val_ds, _class_names, _val_paths = get_datasets(cfg)

    print("📋 Extracting ground-truth labels...")
    y_true = get_true_labels(val_ds, cfg["num_classes"])

    prob_matrices = []
    predict_fn    = predict_with_tta if use_tta else predict_without_tta

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n📦 Loading {model_file}...")

        try:
            # ── FIX: pass actual class objects, not None ─────────────────────
            # Keras needs these to reconstruct custom layers from the saved
            # model config.  Passing None causes:
            #   TypeError: 'str' object is not callable
            # because Keras tries to call None() as a constructor.
            model = keras.models.load_model(
                model_path,
                custom_objects={
                    "Patches":      Patches,
                    "PatchEncoder": PatchEncoder,
                },
                safe_mode=False,
            )
        except Exception as e:
            print(f"   ❌ Could not load {model_file}: {e}")
            print("   Skipping this model.")
            continue

        tta_label = "with TTA" if use_tta else "without TTA"
        print(f"   Running inference {tta_label}...")
        probs = predict_fn(model, val_ds)
        prob_matrices.append(probs)
        print(f"   Done — shape: {probs.shape}")

        del model
        keras.backend.clear_session()

    if not prob_matrices:
        print("\n❌ No models loaded successfully — cannot ensemble.")
        return

    print(f"\n🗳️  Combining predictions via {method} voting...")
    y_pred = soft_vote(prob_matrices) if method == "soft" else hard_vote(
        prob_matrices
    )

    accuracy = np.mean(y_pred == y_true)

    print("\n" + "=" * 50)
    print(f"  Ensemble method : {method} vote")
    print(f"  TTA             : {'yes' if use_tta else 'no'}")
    print(f"  Models combined : {len(prob_matrices)}")
    print(f"  Val accuracy    : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("=" * 50)

    print("\nPer-model baselines:")
    for name, probs in zip(model_files, prob_matrices):
        single_acc = np.mean(np.argmax(probs, axis=1) == y_true)
        print(f"   {name:45s}  {single_acc:.4f}  ({single_acc * 100:.2f}%)")

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
        help="Voting strategy.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable Test-Time Augmentation.",
    )
    args = parser.parse_args()
    run_ensemble(method=args.method, use_tta=args.tta)
