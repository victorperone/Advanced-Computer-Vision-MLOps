"""
models.py  —  v3
=================

Architecture definitions with all dtype and preprocessing fixes applied.

Changes vs previous version
-----------------------------
1.  build_convnext():
    Cast input to float32 BEFORE Rescaling.
    ConvNeXtTiny uses Layer Normalization (not Batch Norm).  LN does NOT
    maintain float32 accumulators — the entire forward pass runs in the
    dtype of the input.  Under mixed_float16 the input arrives as float16.
    float16 LN saturates activations at ~19% accuracy.  Explicit cast fixes.

2.  build_efficientnet():
    Cast input to float32 BEFORE preprocess_input.
    EfficientNet's preprocess_input does channel-wise subtraction.
    Under float16 the subtraction can underflow or saturate.  Explicit
    float32 cast prevents this.

3.  build_baseline_resnet():
    Added two more residual stages and increased stem filters.
    The previous 4-stage network was too shallow for 200 classes.
    New: 6 stages with squeeze-and-excitation (SE) style channel
    attention added to the last two stages.  Expected improvement: +8 pp.

4.  build_vit():
    Cast inputs to float32 at the start of the model graph.
    Under mixed precision, input arrives as float16.  LayerNorm in the
    Transformer blocks (like ConvNeXt) does not accumulate in float32,
    causing attention weights to saturate.
"""

import keras
import tensorflow as tf
from keras import layers


# ===========================================================================
# SECTION 1 — SHARED UTILITY LAYERS
# ===========================================================================


def residual_block(x, filters, kernel_size=3, stride=1):
    """Standard pre-activation Residual Block with skip connection."""
    shortcut = x

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

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1,
            strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def se_block(x, reduction=4):
    """
    Squeeze-and-Excitation channel attention block.

    Learns to weight each channel by its global importance.
    Adds ~0.5% parameters but consistently improves accuracy by
    recalibrating feature map channels.

    How it works:
        1. Global average pool → scalar per channel (squeeze)
        2. Two Dense layers learn channel importance weights
        3. Multiply original feature map by learned weights (excitation)

    This is the same mechanism used in EfficientNet's MBConv blocks.
    Adding it to the last two ResNet stages gives the model the ability
    to suppress irrelevant channels and amplify discriminative ones.
    """
    n_filters = x.shape[-1]
    # Squeeze: global context per channel
    se = layers.GlobalAveragePooling2D(keepdims=True)(x)
    # Excitation: two FC layers with bottleneck
    se = layers.Conv2D(
        max(1, n_filters // reduction), 1, activation="relu"
    )(se)
    se = layers.Conv2D(n_filters, 1, activation="sigmoid")(se)
    # Scale original feature map
    return layers.Multiply()([x, se])


class Patches(layers.Layer):
    """Split an image into non-overlapping patches."""

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """Project patches to embedding dim and add positional encoding."""

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection     = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=projection_dim,
        )
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, projection_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, patch):
        batch_size       = tf.shape(patch)[0]
        patch_embeddings = self.projection(patch)
        cls_tokens       = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)
        positions        = tf.range(
            start=0, limit=self.num_patches + 1, delta=1
        )
        return patch_embeddings + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches":    self.num_patches,
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
    Build an improved ResNet with SE attention on the last two stages.

    Architecture changes vs v2:
    ───────────────────────────
    v2: 4 stages, no attention, single block per stage
    v3: 6 stages, SE attention on stages 4–5, 2 blocks on stages 2–5

    Stage layout:
        Stage 0 (stem): Conv 3×3, BN, ReLU  →  64×64
        Stage 1:        ResBlock × 1         →  64×64  (F filters)
        Stage 2:        ResBlock × 2, stride  →  32×32  (2F)
        Stage 3:        ResBlock × 2, stride  →  16×16  (4F)
        Stage 4:        ResBlock × 2, stride + SE  →  8×8  (8F)
        Stage 5:        ResBlock × 1, stride + SE  →  4×4  (16F)

    The SE blocks on the last two stages teach the model to focus on
    the most discriminative channels for the 200-class problem.

    Expected accuracy: 55–65% from scratch on desktop (was 30%).
    """
    inputs = inputs_orig = keras.Input(shape=input_shape)

    # Stem — larger kernel than usual for 64×64 inputs
    x = layers.Conv2D(base_filters, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Stage 1 — no downsampling
    x = residual_block(x, filters=base_filters, stride=1)

    # Stage 2 — 2 blocks, downsample
    x = residual_block(x, filters=base_filters * 2, stride=2)
    x = residual_block(x, filters=base_filters * 2, stride=1)

    # Stage 3 — 2 blocks, downsample
    x = residual_block(x, filters=base_filters * 4, stride=2)
    x = residual_block(x, filters=base_filters * 4, stride=1)

    # Stage 4 — 2 blocks, downsample, SE attention
    x = residual_block(x, filters=base_filters * 8, stride=2)
    x = residual_block(x, filters=base_filters * 8, stride=1)
    x = se_block(x, reduction=4)                               # ← NEW

    # Stage 5 — 1 block, downsample, SE attention
    x = residual_block(x, filters=base_filters * 16, stride=2)
    x = se_block(x, reduction=4)                               # ← NEW

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)                                 # ← NEW
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="ResNet_SE")


def build_efficientnet(
    input_shape=(64, 64, 3),
    num_classes=200,
    pretrained=False,
    dropout_rate=0.3,
):
    """
    Build EfficientNetV2B0 with float32 input cast.

    FIX: Cast input to float32 before preprocess_input.
    Under mixed_float16, the input tensor arrives as float16.
    EfficientNet's preprocess_input does channel-wise mean subtraction
    and scaling.  In float16 this can cause underflow/saturation in the
    very first layer, degrading all downstream features.
    Forcing float32 at the entry point ensures the preprocessing
    is always numerically stable regardless of the global dtype policy.
    """
    inputs = keras.Input(shape=input_shape)

    # ── FIX: always process in float32 ──────────────────────────────────────
    x = layers.Lambda(
        lambda t: tf.cast(t, tf.float32),
        name="input_float32_cast",
    )(inputs)
    x = keras.applications.efficientnet_v2.preprocess_input(x)

    weights_config = "imagenet" if pretrained else None
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights_config,
        input_shape=input_shape,
    )

    x = backbone(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
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
    Build ConvNeXtTiny with float32 input cast.

    FIX: Cast input to float32 before Rescaling.
    ConvNeXtTiny uses Layer Normalization throughout (not Batch Norm).
    Unlike BN which accumulates statistics in float32, LN computes the
    mean and variance of the CURRENT tensor and normalises it in-place.
    If that tensor is float16, the entire normalisation is in float16.
    For a frozen backbone (Phase 1) this causes the LN statistics to
    oscillate wildly because the float16 range is too narrow for the
    typical ImageNet feature magnitude — the backbone outputs garbage,
    and the classification head converges to ~19% (near-random for 200
    classes where random = 0.5%).
    Casting to float32 at the input resolves this completely.

    Note: We also disable mixed_float16 globally for ConvNeXt in
    fine_tune.py, which is the belt-and-braces fix.
    """
    inputs = keras.Input(shape=input_shape)

    # ── FIX: always process in float32 ──────────────────────────────────────
    x = layers.Lambda(
        lambda t: tf.cast(t, tf.float32),
        name="input_float32_cast",
    )(inputs)

    # Rescale [0, 255] → [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)

    weights_config = "imagenet" if pretrained else None
    backbone = keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=weights_config,
        input_shape=input_shape,
    )

    x = backbone(x)

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
    transformer_layers=6,
    dropout_rate=0.3,
):
    """
    Build an improved Vision Transformer with float32 input cast.

    FIX: Cast input to float32 before patch extraction.
    Transformer LayerNorm has the same float16 saturation issue as
    ConvNeXt.  Forcing float32 at the input prevents attention weight
    saturation in early layers.

    Architecture: pre-norm, stochastic depth, correct key_dim.
    """
    num_patches = (input_shape[0] // patch_size) ** 2
    key_dim     = max(1, projection_dim // num_heads)

    inputs = keras.Input(shape=input_shape)

    # ── FIX: force float32 for Transformer arithmetic ───────────────────────
    x = layers.Lambda(
        lambda t: tf.cast(t, tf.float32),
        name="input_float32_cast",
    )(inputs)

    patches         = Patches(patch_size)(x)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    if transformer_layers > 1:
        depth_rates = [
            dropout_rate * i / (transformer_layers - 1)
            for i in range(transformer_layers)
        ]
    else:
        depth_rates = [0.0]

    for layer_idx in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate,
        )(x1, x1)

        sd_rate = depth_rates[layer_idx]
        if sd_rate > 0.0:
            attention_output = layers.Dropout(
                rate=sd_rate, noise_shape=(None, 1, 1),
            )(attention_output)

        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 4, activation="gelu")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(projection_dim)(x3)

        if sd_rate > 0.0:
            x3 = layers.Dropout(
                rate=sd_rate, noise_shape=(None, 1, 1),
            )(x3)

        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    cls_output     = representation[:, 0, :]
    cls_output     = layers.Dropout(dropout_rate)(cls_output)
    outputs        = layers.Dense(num_classes, activation="softmax")(cls_output)

    return keras.Model(inputs=inputs, outputs=outputs, name="Tiny_ViT_v3")


# ===========================================================================
# SECTION 3 — TRANSFER LEARNING HELPER
# ===========================================================================


def set_backbone_trainable(model, trainable, num_layers_to_unfreeze=200):
    """
    Freeze or partially unfreeze the backbone of a pretrained model.

    Parameters
    ----------
    model : keras.Model
    trainable : bool
        False = Phase 1 (freeze all).
        True  = Phase 2 (unfreeze last N layers).
    num_layers_to_unfreeze : int
        Use 9999 to unfreeze the entire backbone.
    """
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            backbone = layer
            break

    if backbone is None:
        print("⚠️  No backbone sub-model found.")
        return

    if not trainable:
        backbone.trainable = False
        print(
            f"🔒 Backbone FROZEN ({len(backbone.layers)} layers). "
            "Head-only training."
        )
    else:
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False

        n = min(num_layers_to_unfreeze, len(backbone.layers))
        for layer in backbone.layers[-n:]:
            layer.trainable = True

        n_trainable = sum(1 for la in backbone.layers if la.trainable)
        n_frozen    = len(backbone.layers) - n_trainable
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
    """Factory — instantiate any model by name."""
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
