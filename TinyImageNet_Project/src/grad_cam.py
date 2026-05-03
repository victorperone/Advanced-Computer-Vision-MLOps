"""
grad_cam.py
===========

Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability.

Grad-CAM answers the question: "Which pixels caused this prediction?"

It works by:
    1. Running a forward pass and recording the activations of the last
       convolutional layer (the "feature map").
    2. Computing the gradient of the predicted class score with respect
       to those activations.
    3. Global-average-pooling the gradients to get one importance weight
       per channel.
    4. Taking a weighted sum of the feature map channels.
    5. ReLU-ing the result (we only care about features that push the
       class score UP, not down).
    6. Upsampling to the original image size and overlaying as a heatmap.

The result is a heatmap where bright regions are the ones the model was
paying attention to when making its decision.

Why this matters for your project
----------------------------------
* Sanity check: does the model look at the object or the background?
* Debugging: misclassified images often show the model attending to noise.
* Portfolio value: Grad-CAM visualisations are expected in any serious
  CV project writeup.

Supported model types
---------------------
* ResNet   — last conv layer: "activation" (the last ReLU before GAP)
* EfficientNet — last conv layer inside the EfficientNetV2B0 backbone
* ConvNeXt — last conv layer inside the ConvNeXtTiny backbone
* ViT      — Grad-CAM is ill-defined for pure transformers (no spatial
             feature map).  We use "Attention Rollout" for ViT instead,
             which traces attention weights through all layers.

Usage
-----
    # Generate Grad-CAM for a random val image using the best ResNet:
    python -m src.grad_cam --model models/resnet_scratch_best.keras

    # Specify a specific image:
    python -m src.grad_cam \\
        --model models/efficientnet_pretrained_best.keras \\
        --image data/tiny-imagenet-200/val/images/n01443537/val_0.JPEG

PEP 8 notes
-----------
* Max line length 79 characters.
* Two blank lines between top-level definitions.
* Docstrings follow NumPy style.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
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

set_global_seed(cfg.get("seed", 42))


# ===========================================================================
# GRAD-CAM FOR CONVOLUTIONAL MODELS
# ===========================================================================


def find_last_conv_layer(model):
    """
    Walk the model graph backwards to find the last convolutional layer.

    For transfer learning models (EfficientNet, ConvNeXt), the backbone
    is a sub-model.  We recurse into it to find the true last conv.

    Parameters
    ----------
    model : keras.Model

    Returns
    -------
    str
        The name of the last Conv2D or DepthwiseConv2D layer found.

    Raises
    ------
    ValueError
        If no convolutional layer is found in the model.
    """
    # Walk layers in reverse order — first Conv2D we find going backwards
    # is the last one in the forward pass.
    for layer in reversed(model.layers):
        # Sub-models (EfficientNet backbone) — recurse
        if hasattr(layer, "layers"):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue

        # Check the layer type
        if isinstance(
            layer,
            (
                keras.layers.Conv2D,
                keras.layers.DepthwiseConv2D,
                keras.layers.SeparableConv2D,
            ),
        ):
            return layer.name

    raise ValueError("No convolutional layer found in model.")


def compute_gradcam(model, image, class_idx=None, layer_name=None):
    """
    Compute a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model : keras.Model
        Trained convolutional model.
    image : np.ndarray, shape (H, W, C)
        Single image, already normalised (same preprocessing as training).
    class_idx : int or None
        Which class to explain.  If None, uses the model's top-1 prediction.
    layer_name : str or None
        Which convolutional layer to use.  If None, auto-detects the last one.

    Returns
    -------
    heatmap : np.ndarray, shape (H, W)
        Float32 array in [0, 1] — the Grad-CAM saliency map at image resolution.
    pred_class : int
        The class index that was explained (the argmax prediction if
        class_idx was None).
    pred_confidence : float
        The model's confidence in ``pred_class`` (softmax probability).
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
        print(f"   Using last conv layer: '{layer_name}'")

    # Build a sub-model that outputs BOTH the target conv layer activations
    # AND the final softmax predictions simultaneously.
    # This is more efficient than two separate forward passes.
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,  # Feature map
            model.output,                         # Final predictions
        ],
    )

    # Add batch dimension: (H, W, C) → (1, H, W, C)
    img_batch = tf.expand_dims(image, axis=0)
    img_batch = tf.cast(img_batch, tf.float32)

    # Use GradientTape to record gradients of the class score w.r.t. the
    # feature map activations.
    with tf.GradientTape() as tape:
        # Watch the input so tape tracks the computation graph through it
        tape.watch(img_batch)

        # Forward pass — record activations and predictions
        conv_outputs, predictions = grad_model(img_batch)

        # Which class are we explaining?
        if class_idx is None:
            class_idx = tf.argmax(predictions[0]).numpy()

        pred_confidence = float(predictions[0, class_idx])

        # The "score" we differentiate is the raw logit for our class.
        # Using the logit (before softmax) gives cleaner gradients.
        class_score = predictions[:, class_idx]

    # Gradient of the class score w.r.t. every activation in the conv layer.
    # Shape: (1, feature_h, feature_w, num_channels)
    grads = tape.gradient(class_score, conv_outputs)

    # Global average pooling of gradients across spatial dimensions →
    # one importance weight per channel.
    # Shape: (1, 1, 1, num_channels)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2), keepdims=True)

    # Weight the feature map channels by their importance.
    # Shape: (1, feature_h, feature_w, num_channels)
    weighted_map = conv_outputs * pooled_grads

    # Sum across channels to get a single spatial heatmap.
    # Shape: (feature_h, feature_w)
    heatmap = tf.reduce_sum(weighted_map, axis=-1)[0]

    # ReLU: keep only features that INCREASE the class score.
    # Negative values mean "this feature suppresses the class" — not helpful.
    heatmap = tf.nn.relu(heatmap)

    # Normalise to [0, 1] for visualisation.
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Upsample to original image spatial resolution
    img_h, img_w = image.shape[0], image.shape[1]
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],   # Add channel dim for resize
        [img_h, img_w],
    ).numpy()[..., 0]               # Remove channel dim

    return heatmap_resized, int(class_idx), pred_confidence


# ===========================================================================
# ATTENTION ROLLOUT FOR VIT
# ===========================================================================


def compute_attention_rollout(model, image):
    """
    Compute Attention Rollout for a Vision Transformer model.

    Standard Grad-CAM does not apply to ViT because there are no spatial
    feature maps — only a flat sequence of patch tokens.  Instead, we use
    Attention Rollout (Abnar & Zuidema, 2020):

        1. Extract the attention weight matrices from every transformer layer.
        2. Add the identity matrix to each (to model the residual connection,
           which adds "self-attention" to the skip path).
        3. Normalise each matrix row-wise.
        4. Multiply them together in sequence — this propagates attention
           through the depth of the network.
        5. The first row (corresponding to the [CLS] token) of the product
           tells us how much the CLS token "attended to" each patch position.

    The resulting vector has one value per patch; we reshape it back to a
    spatial grid and upsample to the image size.

    Parameters
    ----------
    model : keras.Model
        A ViT model with the architecture from models.py.
    image : np.ndarray, shape (H, W, C)
        Single normalised image.

    Returns
    -------
    heatmap : np.ndarray, shape (H, W)
        Attention rollout saliency map at image resolution.
    pred_class : int
        Top-1 predicted class.
    pred_confidence : float
        Confidence in the prediction.
    """
    img_batch = tf.expand_dims(image, axis=0).numpy().astype(np.float32)

    # --- Find all MultiHeadAttention layers ---
    attention_layers = [
        layer for layer in model.layers
        if isinstance(layer, keras.layers.MultiHeadAttention)
    ]

    if len(attention_layers) == 0:
        raise ValueError(
            "No MultiHeadAttention layers found. "
            "Is this really a ViT model?"
        )

    # Build a model that outputs attention weights from every MHA layer.
    # MultiHeadAttention.call() returns (output, attention_weights) when
    # return_attention_scores=True, but this is hard to extract post-hoc.
    # Simpler approach: call attention layers directly on the encoded patches.

    # Run the full model to get the prediction
    predictions = model(img_batch, training=False)
    pred_class = int(tf.argmax(predictions[0]))
    pred_confidence = float(predictions[0, pred_class])

    # --- Approximate rollout from the model's attention sub-models ---
    # Because extracting intermediate attention weights from a Keras model
    # graph requires custom layer registration, we use a simplified version:
    # we extract attention scores by running the MHA layers directly.

    # Get the PatchEncoder output (the token sequence before transformer blocks)
    # We walk the model to find the PatchEncoder layer
    patch_encoder = None
    patches_layer = None
    for layer in model.layers:
        if "patch_encoder" in layer.name.lower():
            patch_encoder = layer
        if "patches" in layer.name.lower():
            patches_layer = layer

    if patch_encoder is None:
        print(
            "⚠️  Could not find PatchEncoder layer — "
            "returning uniform heatmap."
        )
        img_size = image.shape[0]
        return np.ones((img_size, img_size)), pred_class, pred_confidence

    # Forward pass through the patch extraction and encoding stages
    patches_output = patches_layer(img_batch)
    encoded = patch_encoder(patches_output)

    # Collect attention matrices from each transformer block
    rollout = None
    num_tokens = encoded.shape[1]   # num_patches + 1 (for [CLS])

    for attn_layer in attention_layers:
        # Call each MHA layer and request attention scores
        _, attn_weights = attn_layer(
            encoded, encoded,
            return_attention_scores=True,
            training=False,
        )
        # attn_weights: (1, num_heads, seq_len, seq_len)
        # Average across heads
        attn = tf.reduce_mean(attn_weights[0], axis=0).numpy()

        # Add identity (residual connection) and normalise rows
        attn = attn + np.eye(num_tokens)
        attn = attn / attn.sum(axis=-1, keepdims=True)

        if rollout is None:
            rollout = attn
        else:
            rollout = rollout @ attn

    if rollout is None:
        img_size = image.shape[0]
        return np.ones((img_size, img_size)), pred_class, pred_confidence

    # Row 0 = [CLS] token attending to all patches; skip col 0 ([CLS] self)
    cls_attention = rollout[0, 1:]   # shape: (num_patches,)

    # Reshape to spatial grid
    grid_size = int(np.sqrt(len(cls_attention)))
    attn_map = cls_attention.reshape(grid_size, grid_size)

    # Upsample to image size
    img_h, img_w = image.shape[0], image.shape[1]
    heatmap = tf.image.resize(
        attn_map[..., np.newaxis],
        [img_h, img_w],
    ).numpy()[..., 0]

    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, pred_class, pred_confidence


# ===========================================================================
# VISUALISATION
# ===========================================================================


def overlay_heatmap(image_raw, heatmap, alpha=0.45):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    image_raw : np.ndarray, shape (H, W, C)
        Original image in [0, 255] uint8 range (for display).
    heatmap : np.ndarray, shape (H, W)
        Normalised saliency map in [0, 1].
    alpha : float
        Blend factor: 0 = only image, 1 = only heatmap.

    Returns
    -------
    np.ndarray, shape (H, W, 3)
        Blended RGB image suitable for plt.imshow().
    """
    import matplotlib.cm as cm

    # Map heatmap values to RGBA colours using the "jet" colormap
    colormap = cm.get_cmap("jet")
    heatmap_coloured = colormap(heatmap)[..., :3]   # Drop alpha channel

    # Normalise original image to [0, 1] for blending
    img_norm = image_raw.astype(np.float32) / 255.0
    if img_norm.max() <= 1.01 and img_norm.min() >= -3.0:
        # Image might be standardised (negative values) — rescale for display
        img_norm = (img_norm - img_norm.min()) / (
            img_norm.max() - img_norm.min() + 1e-8
        )

    overlaid = (1 - alpha) * img_norm + alpha * heatmap_coloured
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid


def run_gradcam(model_path, image_path=None, num_random=5):
    """
    Load a model, pick images, compute saliency maps, and save a figure.

    Parameters
    ----------
    model_path : str
        Path to the ``.keras`` model file.
    image_path : str or None
        Path to a specific image.  If None, samples ``num_random`` random
        images from the validation set.
    num_random : int
        How many random val images to visualise when image_path is None.
    """
    print(f"\n📦 Loading model: {os.path.basename(model_path)}")
    model = keras.models.load_model(
        model_path,
        safe_mode=False,
    )

    # Detect model type from filename
    is_vit = "vit" in os.path.basename(model_path).lower()
    is_pretrained = "pretrained" in os.path.basename(model_path).lower()

    # Collect images to visualise
    if image_path is not None:
        image_paths = [image_path]
    else:
        # Walk the val directory and sample randomly
        val_dir = cfg["val_dir"]
        all_images = []
        for root, _dirs, files in os.walk(val_dir):
            for f in files:
                if f.lower().endswith((".jpeg", ".jpg", ".png")):
                    all_images.append(os.path.join(root, f))

        rng = np.random.default_rng(cfg.get("seed", 42))
        image_paths = list(
            rng.choice(all_images, size=min(num_random, len(all_images)),
                       replace=False)
        )

    img_size = cfg["img_size"]
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    # Handle single-image case where axes is 1-D
    if n == 1:
        axes = [axes]

    for row_idx, img_path in enumerate(image_paths):
        # Load and resize the original image
        raw_img = tf.io.read_file(img_path)
        raw_img = tf.image.decode_image(raw_img, channels=3)
        raw_img = tf.image.resize(raw_img, [img_size, img_size])
        raw_img = raw_img.numpy().astype(np.uint8)

        # Preprocess for the model (same as training pipeline)
        if is_pretrained:
            # EfficientNet / ConvNeXt embed their own preprocessing;
            # feed raw [0, 255] float32
            model_input = raw_img.astype(np.float32)
        else:
            # ViT and ResNet use ImageNet standardisation
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            model_input = (raw_img.astype(np.float32) / 255.0 - mean) / std

        # Compute saliency
        if is_vit:
            heatmap, pred_class, confidence = compute_attention_rollout(
                model, model_input
            )
            method_label = "Attention Rollout"
        else:
            heatmap, pred_class, confidence = compute_gradcam(
                model, model_input
            )
            method_label = "Grad-CAM"

        # Derive class name from directory structure (val/images/n01234/img.jpg)
        parts = img_path.split(os.sep)
        class_folder = parts[-2] if len(parts) >= 2 else "unknown"

        # --- Plot row ---
        ax_orig, ax_heat, ax_blend = axes[row_idx]

        ax_orig.imshow(raw_img)
        ax_orig.set_title(f"Original\nClass: {class_folder}", fontsize=10)
        ax_orig.axis("off")

        ax_heat.imshow(heatmap, cmap="jet")
        ax_heat.set_title(
            f"{method_label}\nPred: {pred_class} ({confidence:.1%})", fontsize=10
        )
        ax_heat.axis("off")

        blended = overlay_heatmap(raw_img, heatmap, alpha=0.45)
        ax_blend.imshow(blended)
        ax_blend.set_title("Overlay", fontsize=10)
        ax_blend.axis("off")

    plt.suptitle(
        f"Saliency maps — {os.path.basename(model_path)}",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    # Save to logs/evaluation/<model_name>/
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    out_dir = os.path.join(cfg["logs_dir"], "evaluation", model_stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gradcam_samples.png")

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saliency map saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM / Attention Rollout saliency maps."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the .keras model file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a specific image.  If omitted, samples 5 random val images.",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=5,
        help="Number of random val images when --image is not specified.",
    )
    args = parser.parse_args()

    run_gradcam(
        model_path=args.model,
        image_path=args.image,
        num_random=args.num_random,
    )
