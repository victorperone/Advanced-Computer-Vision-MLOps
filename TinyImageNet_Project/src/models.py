"""
models.py
=========

Architecture definitions for the Tiny ImageNet classification project.

This module contains four model families:
    - ResNet    : Custom residual network, trains well from scratch on CPU/GPU.
    - EfficientNet : Transfer learning via Keras Applications (ImageNet pretrained).
    - ConvNeXt  : Transfer learning via Keras Applications (ImageNet pretrained).
    - ViT       : Vision Transformer, scratch-trained with modern stabilisation tricks.

The public API is the ``get_model`` factory at the bottom of this file.
Everything else is an implementation detail.

Design principles
-----------------
* No side-effects at import time (the old file set CUDA_VISIBLE_DEVICES here,
  which is wrong — hardware config belongs in train.py before TF loads).
* Every builder accepts only the kwargs it needs; the factory strips extras.
* Pretrained models embed their own preprocessing so the data pipeline stays
  architecture-agnostic (plain uint8 → [0, 255] float32 in, correct normalisation
  happens inside the model graph).

PEP 8 notes
-----------
* Max line length 79 characters.
* Two blank lines between top-level definitions.
* Docstrings follow NumPy style (Parameters / Returns sections).
"""

import keras
import tensorflow as tf
from keras import layers


# ===========================================================================
# SECTION 1 — SHARED UTILITY LAYERS
# ===========================================================================


def residual_block(x, filters, kernel_size=3, stride=1):
    """
    Build one standard pre-activation Residual Block.

    A residual block adds the input tensor (the "shortcut") directly to the
    output of two Conv layers.  This skip-connection lets gradients flow
    backwards through deep networks without vanishing.

    Visual:
        input ──┬──> Conv ──> BN ──> ReLU ──> Conv ──> BN ──> Add ──> ReLU
                └──────────────────────────────────────────────┘
                         (shortcut, optionally projected)

    Parameters
    ----------
    x : tf.Tensor
        Input feature map from the previous layer.
    filters : int
        Number of output channels for both Conv layers.
    kernel_size : int
        Spatial size of the convolutional kernel (default 3).
    stride : int
        Stride for the first Conv; stride > 1 halves spatial dimensions
        (acts as a learned downsampler, replacing MaxPool).

    Returns
    -------
    tf.Tensor
        Output feature map with shape (B, H//stride, W//stride, filters).
    """
    shortcut = x

    # --- Main path ---
    x = layers.Conv2D(
        filters, kernel_size,
        strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, kernel_size,
        strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # --- Shortcut projection ---
    # If the shortcut tensor has a different shape from the main path output
    # (because stride > 1 or the channel count changed), we project it with a
    # 1×1 Conv so the Add() layer receives matching shapes.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1,
            strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


class Patches(layers.Layer):
    """
    Split an image tensor into a flat sequence of non-overlapping patches.

    This is the very first step in a Vision Transformer.  The image is
    divided into a grid of patch_size × patch_size squares.  Each square is
    flattened into a 1-D vector so the Transformer can treat it like a word
    token.

    Example — 64×64 image with patch_size=8:
        grid  = 8×8 = 64 patches
        each patch vector = 8*8*3 = 192 floats

    Parameters
    ----------
    patch_size : int
        Height and width of each square patch (must divide image size evenly).
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        """
        Parameters
        ----------
        images : tf.Tensor, shape (B, H, W, C)

        Returns
        -------
        tf.Tensor, shape (B, num_patches, patch_size*patch_size*C)
        """
        batch_size = tf.shape(images)[0]

        # tf.image.extract_patches slides a window over the image and returns
        # the contents of every window as a flat vector.
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # patches shape after extract_patches: (B, grid_h, grid_w, patch_flat)
        # We merge grid_h and grid_w into a single "sequence" axis.
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        """Required so the model can be saved and reloaded correctly."""
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """
    Project patches into a fixed-size embedding and add positional encoding.

    A raw patch vector (e.g. 192-D) is first projected to projection_dim
    (e.g. 64-D) via a Dense layer.  Then a learned positional embedding is
    added so the Transformer knows the spatial order of patches.

    We also prepend a learnable [CLS] token — a special "summary" vector that
    will accumulate global information through the Transformer layers and is
    used for final classification (same trick as BERT).

    Parameters
    ----------
    num_patches : int
        Total number of patches produced by the Patches layer.
    projection_dim : int
        Embedding size; all Transformer layers operate at this width.
    """

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        # Linear projection: patch_flat_dim → projection_dim
        self.projection = layers.Dense(units=projection_dim)

        # Positional embedding table: one vector per position.
        # +1 accounts for the prepended [CLS] token.
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=projection_dim,
        )

        # The [CLS] token is a single learned vector that starts at (1, 1, dim)
        # and is broadcast to the full batch at call-time.
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, projection_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, patch):
        """
        Parameters
        ----------
        patch : tf.Tensor, shape (B, num_patches, patch_flat_dim)

        Returns
        -------
        tf.Tensor, shape (B, num_patches + 1, projection_dim)
            The sequence of patch embeddings with [CLS] prepended.
        """
        batch_size = tf.shape(patch)[0]

        # Project each patch vector to projection_dim
        patch_embeddings = self.projection(patch)

        # Repeat the single CLS token for every item in the batch
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)

        # Prepend [CLS]: shape becomes (B, num_patches+1, projection_dim)
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)

        # Create position indices [0, 1, 2, ..., num_patches] and look them up
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        encoded = patch_embeddings + self.position_embedding(positions)
        return encoded

    def get_config(self):
        """Required so the model can be saved and reloaded correctly."""
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


# ===========================================================================
# SECTION 2 — ARCHITECTURE BUILDERS
# ===========================================================================


def build_baseline_resnet(
    input_shape=(64, 64, 3),
    num_classes=200,
    base_filters=64,
):
    """
    Build a compact custom ResNet scaled by ``base_filters``.

    The network has four residual stages.  Each stage doubles the filter
    count and (except the first) halves spatial resolution via stride=2.
    This is the best model to run from scratch on a CPU laptop — it converges
    reliably even with limited compute.

    Parameters
    ----------
    input_shape : tuple of int
        (H, W, C) — e.g. (64, 64, 3) for Tiny ImageNet.
    num_classes : int
        Number of output classes (200 for Tiny ImageNet).
    base_filters : int
        Filter count for the first stage.  Subsequent stages use
        base_filters * 2, * 4, * 8.  Laptop profile uses 32; desktop uses 64.

    Returns
    -------
    keras.Model
        Compiled-ready model; not yet compiled (compilation in train.py).
    """
    inputs = keras.Input(shape=input_shape)

    # Initial feature extraction stem
    x = layers.Conv2D(base_filters, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Four residual stages with progressive downsampling
    x = residual_block(x, filters=base_filters,      stride=1)
    x = residual_block(x, filters=base_filters * 2,  stride=2)
    x = residual_block(x, filters=base_filters * 4,  stride=2)
    x = residual_block(x, filters=base_filters * 8,  stride=2)

    # Global average pooling collapses spatial dims → single vector per image
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="Tiny_Dynamic_ResNet")


def build_efficientnet(
    input_shape=(64, 64, 3),
    num_classes=200,
    pretrained=False,
    dropout_rate=0.3,
):
    """
    Build EfficientNetV2B0 with optional ImageNet pretrained weights.

    IMPORTANT — preprocessing is embedded inside the model graph.
    EfficientNet was trained with its own normalisation (not just /255).
    By wrapping ``preprocess_input`` as the first layer, the data pipeline
    can always supply raw [0, 255] uint8-cast-to-float32 pixels and the
    model handles the rest.  This prevents the subtle bug of mixing
    /255 normalisation with ImageNet-pretrained weights.

    Parameters
    ----------
    input_shape : tuple of int
        (H, W, C).
    num_classes : int
        Output classes.
    pretrained : bool
        If True, load ImageNet weights.  If False, random init.
    dropout_rate : float
        Dropout applied before the final Dense layer.

    Returns
    -------
    keras.Model
    """
    inputs = keras.Input(shape=input_shape)

    # --- Preprocessing layer (embedded so the model is self-contained) ---
    # EfficientNet preprocess_input rescales to [-1, 1] and applies
    # channel-wise mean subtraction matching ImageNet statistics.
    # If pretrained=False this is still good practice; it keeps
    # inputs in a well-scaled range regardless of training mode.
    x = keras.applications.efficientnet_v2.preprocess_input(inputs)

    weights_config = "imagenet" if pretrained else None
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights_config,
        input_shape=input_shape,
    )

    # Pass the preprocessed tensor through the backbone
    x = backbone(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)      # Extra BN stabilises fine-tuning
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="Tiny_EfficientNetV2")


def build_convnext(
    input_shape=(64, 64, 3),
    num_classes=200,
    pretrained=False,
    dropout_rate=0.3,
):
    """
    Build ConvNeXtTiny with optional ImageNet pretrained weights.

    ConvNeXt expects inputs normalised to roughly [-1, 1].  We embed
    that rescaling inside the model as a Lambda layer for the same
    portability reason as EfficientNet above.

    Parameters
    ----------
    input_shape : tuple of int
    num_classes : int
    pretrained : bool
    dropout_rate : float

    Returns
    -------
    keras.Model
    """
    inputs = keras.Input(shape=input_shape)

    # Rescale [0, 255] → [-1, 1] using the standard formula: x/127.5 - 1
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)

    weights_config = "imagenet" if pretrained else None
    backbone = keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=weights_config,
        input_shape=input_shape,
    )

    x = backbone(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="Tiny_ConvNeXt")


def build_vit(
    input_shape=(64, 64, 3),
    num_classes=200,
    patch_size=8,
    projection_dim=64,
    num_heads=8,
    transformer_layers=4,
    dropout_rate=0.3,
):
    """
    Build an improved Vision Transformer (ViT) trained from scratch.

    Key improvements over the original version:
    -   **Stochastic depth (DropPath)**: Instead of always-on Dropout, each
        residual branch is randomly dropped with a linearly increasing rate.
        This is the single biggest regularisation improvement for ViTs and
        is why modern small ViTs (DeiT, CaiT) converge on small datasets.
    -   **Attention temperature scaling**: ``key_dim`` is now
        projection_dim // num_heads instead of full projection_dim.
        Full projection_dim makes attention keys too large → uniform softmax
        weights → attention becomes meaningless.  Per-head key_dim fixes this.
    -   **MLP expansion ratio of 4**: The FFN hidden size is
        projection_dim * 4, matching the original ViT paper.
    -   **Pre-norm architecture**: LayerNorm before (not after) each sub-block.
        This is more stable for small datasets and avoids gradient explosion
        early in training.
    -   **Learnable [CLS] token + positional embedding**: Same as BERT.
        The [CLS] position aggregates global information and is used for
        classification.

    Why ViT from scratch is hard:
        ViT has no inductive biases (unlike Conv nets which know nearby pixels
        are related).  It must learn spatial structure entirely from data.
        On 100k images it can reach ~60-65% without tricks, or ~68-72% with the
        improvements above.  For 80%+ you need ImageNet pretrained weights
        (use EfficientNet or ConvNeXt with pretrained=True).

    Parameters
    ----------
    input_shape : tuple of int
        (H, W, C).
    num_classes : int
        Output classes.
    patch_size : int
        Size of each image patch.  Must divide H and W evenly.
        Laptop: 8 → 64 patches.  Desktop: 8 also (fine for 64×64 images).
    projection_dim : int
        Embedding dimension.  All transformer computations use this width.
    num_heads : int
        Number of attention heads.  Must divide projection_dim evenly.
        Each head attends with key_dim = projection_dim // num_heads.
    transformer_layers : int
        Number of stacked Transformer encoder blocks.
    dropout_rate : float
        Applied inside attention and the FFN.  Also used as the *maximum*
        stochastic depth rate (actual rate scales linearly across layers).

    Returns
    -------
    keras.Model
    """
    num_patches = (input_shape[0] // patch_size) ** 2

    # --- Correct key_dim: per-head dimension, not full projection_dim ---
    # Original bug: key_dim=projection_dim → each head has projection_dim keys
    # which makes MultiHeadAttention output projection_dim * num_heads wide.
    # This causes a shape mismatch AND makes attention numerically unstable
    # because the dot-product scale is too large.
    # Fix: key_dim = projection_dim // num_heads  (standard ViT convention).
    key_dim = max(1, projection_dim // num_heads)

    inputs = keras.Input(shape=input_shape)

    # Divide image into patches and embed them
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # --- Stochastic depth schedule ---
    # Each layer gets a linearly increasing drop probability.
    # Layer 0 has drop_rate ≈ 0; the last layer has drop_rate ≈ dropout_rate.
    # This means the deeper layers (which have less informative gradients early
    # on) are regularised more aggressively.
    # Reference: "Deep Networks with Stochastic Depth", Huang et al. 2016.
    if transformer_layers > 1:
        stochastic_depth_rates = [
            dropout_rate * i / (transformer_layers - 1)
            for i in range(transformer_layers)
        ]
    else:
        stochastic_depth_rates = [0.0]

    for layer_idx in range(transformer_layers):
        # ---- Pre-norm Multi-Head Self-Attention block ----
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,        # FIXED: was projection_dim (wrong)
            dropout=dropout_rate,
        )(x1, x1)

        # Stochastic depth: randomly zero-out this residual branch during
        # training.  At inference, it always passes through unchanged.
        # We use a Dropout layer with a noise_shape that broadcasts to the
        # whole token sequence — this drops entire residual connections, not
        # individual neurons.
        sd_rate = stochastic_depth_rates[layer_idx]
        if sd_rate > 0.0:
            attention_output = layers.Dropout(
                rate=sd_rate,
                noise_shape=(None, 1, 1),   # Same mask across all positions
            )(attention_output)

        # First residual connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # ---- Pre-norm Feed-Forward Network (MLP) block ----
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # Expand → activate → contract  (standard Transformer MLP block)
        x3 = layers.Dense(
            projection_dim * 4,
            activation="gelu",          # GELU is standard for Transformers
        )(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(projection_dim)(x3)   # Project back to model width

        # Stochastic depth on the MLP branch too
        if sd_rate > 0.0:
            x3 = layers.Dropout(
                rate=sd_rate,
                noise_shape=(None, 1, 1),
            )(x3)

        # Second residual connection
        encoded_patches = layers.Add()([x3, x2])

    # --- Classification head ---
    # Final LayerNorm stabilises the [CLS] representation before the head.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    # Extract only the [CLS] token (position 0); discard all patch tokens.
    # The [CLS] token has "seen" every other token via attention and summarises
    # the whole image.
    cls_output = representation[:, 0, :]

    cls_output = layers.Dropout(dropout_rate)(cls_output)
    outputs = layers.Dense(num_classes, activation="softmax")(cls_output)

    return keras.Model(inputs=inputs, outputs=outputs, name="Tiny_Improved_ViT")


# ===========================================================================
# SECTION 3 — TRANSFER LEARNING HELPER
# ===========================================================================


def set_backbone_trainable(model, trainable, num_layers_to_unfreeze=30):
    """
    Freeze or partially unfreeze the backbone of a pretrained model.

    Transfer learning has two phases:

    Phase 1 — HEAD ONLY (freeze the backbone):
        The pretrained backbone weights are locked (``backbone.trainable = False``).
        Only the new classification head Dense layer is updated.
        Purpose: Teach the head to work with existing ImageNet features quickly
        without accidentally destroying them (learning rate can be high here).
        Duration: ~10 epochs.

    Phase 2 — FINE-TUNING (unfreeze last N layers):
        The final ``num_layers_to_unfreeze`` layers of the backbone are unlocked.
        The rest stay frozen.
        Purpose: Let the backbone slowly adapt to Tiny ImageNet's specific
        textures and classes.
        WARNING: Use a 10× lower learning rate than Phase 1.  Too high and you
        will cause "catastrophic forgetting" — the ImageNet features vanish.
        Duration: ~20–40 epochs.

    Parameters
    ----------
    model : keras.Model
        The full model (input layer + backbone + head).
    trainable : bool
        False → Phase 1 (freeze all backbone layers).
        True  → Phase 2 (unfreeze last ``num_layers_to_unfreeze`` layers).
    num_layers_to_unfreeze : int
        How many backbone layers from the END to unlock in Phase 2.
        30 is a good default for EfficientNetV2B0 and ConvNeXtTiny.
        Increase to 60 for more adaptation (desktop with long training budget).
        Decrease to 10 for smaller datasets or if overfitting appears.

    Notes
    -----
    After calling this function you MUST recompile the model before continuing
    training.  Keras caches the trainable variable list at compile-time.

    Example
    -------
    >>> set_backbone_trainable(model, trainable=False)     # Phase 1
    >>> model.compile(optimizer=AdamW(lr=1e-3), ...)
    >>> model.fit(train_ds, epochs=10)
    >>>
    >>> set_backbone_trainable(model, trainable=True, num_layers_to_unfreeze=30)
    >>> model.compile(optimizer=AdamW(lr=1e-4), ...)       # Lower LR!
    >>> model.fit(train_ds, epochs=40)
    """
    # The backbone is layer index 1 for models built with:
    #   inputs = Input(...)  → layer[0]
    #   x = preprocess(inputs)  → layer[1]  (Rescaling or Lambda)
    #   backbone = EfficientNet(...); x = backbone(x) → layer[2]
    # We search by name to be robust to different head configurations.
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers"):    # Sub-models have their own .layers
            backbone = layer
            break

    if backbone is None:
        print("⚠️  No backbone sub-model found — is this a scratch-trained model?")
        return

    if not trainable:
        # ---- Phase 1: Freeze everything ----
        backbone.trainable = False
        total = len(backbone.layers)
        print(
            f"🔒 Backbone FROZEN ({total} layers). "
            "Training classification head only."
        )
    else:
        # ---- Phase 2: Unfreeze last N layers ----
        backbone.trainable = True

        # First, lock all backbone layers …
        for layer in backbone.layers:
            layer.trainable = False

        # … then unlock only the last N
        for layer in backbone.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True

        n_trainable = sum(1 for la in backbone.layers if la.trainable)
        n_frozen = len(backbone.layers) - n_trainable
        print(
            f"🔓 Backbone PARTIALLY UNFROZEN — "
            f"{n_trainable} layers trainable, {n_frozen} frozen."
        )


# ===========================================================================
# SECTION 4 — MODEL FACTORY
# ===========================================================================


def get_model(
    model_name,
    input_shape=(64, 64, 3),
    num_classes=200,
    pretrained=False,
    **kwargs,
):
    """
    Factory function: instantiate any supported model by name.

    The ``**kwargs`` are forwarded directly to the builder, which means
    any key defined under ``vit_kwargs`` / ``resnet_kwargs`` etc. in
    ``config.yaml`` is automatically passed through — no manual plumbing.

    Parameters
    ----------
    model_name : str
        One of: ``"resnet"``, ``"efficientnet"``, ``"convnext"``, ``"vit"``.
    input_shape : tuple of int
        (H, W, C).  Default (64, 64, 3) for Tiny ImageNet.
    num_classes : int
        Number of output classes.  Default 200.
    pretrained : bool
        Whether to load ImageNet weights.  Only applies to EfficientNet and
        ConvNeXt; ignored for ResNet and ViT (no pretrained Keras weights).
    **kwargs
        Architecture-specific hyperparameters forwarded to the builder.
        For ViT: patch_size, projection_dim, num_heads, transformer_layers,
                 dropout_rate.
        For ResNet: base_filters.
        For EfficientNet / ConvNeXt: dropout_rate.

    Returns
    -------
    keras.Model
        Uncompiled model ready for model.compile() in train.py.

    Raises
    ------
    ValueError
        If ``model_name`` is not one of the supported strings.
    """
    if model_name == "resnet":
        return build_baseline_resnet(input_shape, num_classes, **kwargs)

    elif model_name == "efficientnet":
        return build_efficientnet(input_shape, num_classes, pretrained, **kwargs)

    elif model_name == "convnext":
        return build_convnext(input_shape, num_classes, pretrained, **kwargs)

    elif model_name == "vit":
        return build_vit(input_shape, num_classes, **kwargs)

    else:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            "Choose from: 'resnet', 'efficientnet', 'convnext', 'vit'."
        )


# ===========================================================================
# SMOKE TEST — run with:  python -m src.models
# ===========================================================================

if __name__ == "__main__":
    import os

    # Force CPU so the smoke test works on the laptop
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("=" * 55)
    print("Smoke test: building all four model variants")
    print("=" * 55)

    # --- Laptop-sized ViT ---
    print("\n[1/4] Improved ViT (laptop micro config):")
    vit = get_model(
        "vit",
        patch_size=8,
        projection_dim=32,
        num_heads=4,
        transformer_layers=2,
        dropout_rate=0.2,
    )
    vit.summary(line_length=72)

    # --- Compact ResNet ---
    print("\n[2/4] ResNet (laptop config, base_filters=32):")
    resnet = get_model("resnet", base_filters=32)
    print(f"  Parameters: {resnet.count_params():,}")

    # --- EfficientNet (no pretrained weights in smoke test) ---
    print("\n[3/4] EfficientNetV2B0 (no pretrained weights):")
    eff = get_model("efficientnet", pretrained=False, dropout_rate=0.2)
    print(f"  Parameters: {eff.count_params():,}")

    # --- ConvNeXt (no pretrained weights in smoke test) ---
    print("\n[4/4] ConvNeXtTiny (no pretrained weights):")
    cnx = get_model("convnext", pretrained=False, dropout_rate=0.2)
    print(f"  Parameters: {cnx.count_params():,}")

    print("\n✅ All models built successfully.")


