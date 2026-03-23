import os
# Hide GPUs to force CPU execution on the laptop
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras import layers
import tensorflow as tf

# ==========================================
# 1. UTILITY FUNCTIONS & CUSTOM LAYERS
# ==========================================

def residual_block(x, filters, kernel_size=3, stride=1):
    """A standard Residual Block with a skip connection."""
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

class Patches(layers.Layer):
    """Physically chops the image into a grid of squares and flattens them."""
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

class PatchEncoder(layers.Layer):
    """Projects patches and adds learned Positional Embeddings."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, 
            output_dim=projection_dim
        )
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, projection_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        patch_embeddings = self.projection(patch)
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        encoded = patch_embeddings + self.position_embedding(positions)
        return encoded


# ==========================================
# 2. DYNAMIC ARCHITECTURE BUILDERS
# ==========================================

def build_baseline_resnet(input_shape=(64, 64, 3), num_classes=200, base_filters=64):
    """Builds a dynamic ResNet controlled by base_filters from the YAML."""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(base_filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, filters=base_filters, stride=1)      
    x = residual_block(x, filters=base_filters * 2, stride=2)  
    x = residual_block(x, filters=base_filters * 4, stride=2)  
    x = residual_block(x, filters=base_filters * 8, stride=2)  

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs, name="Tiny_Dynamic_ResNet")

def build_efficientnet(input_shape=(64, 64, 3), num_classes=200, pretrained=False, dropout_rate=0.3):
    inputs = keras.Input(shape=input_shape)
    weights_config = "imagenet" if pretrained else None
    
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False, weights=weights_config,
        input_tensor=inputs, input_shape=input_shape
    )
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name="Tiny_EfficientNetV2")

def build_convnext(input_shape=(64, 64, 3), num_classes=200, pretrained=False, dropout_rate=0.3):
    inputs = keras.Input(shape=input_shape)
    weights_config = "imagenet" if pretrained else None
    
    base_model = keras.applications.ConvNeXtTiny(
        include_top=False, weights=weights_config,
        input_tensor=inputs, input_shape=input_shape
    )
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
        
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="Tiny_ConvNeXt")

def build_vit(input_shape=(64, 64, 3), num_classes=200, patch_size=8, projection_dim=64, num_heads=8, transformer_layers=4, dropout_rate=0.3):
    """Builds a fully dynamic Vision Transformer from YAML hyperparameters."""
    num_patches = (input_shape[0] // patch_size) ** 2 
    
    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )(x1, x1)
        
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        x3 = layers.Dense(projection_dim * 4, activation="gelu")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        
        encoded_patches = layers.Add()([x3, x2])
        
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = representation[:, 0, :]
    representation = layers.Dropout(dropout_rate)(representation)
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    return keras.Model(inputs=inputs, outputs=outputs, name="Tiny_Dynamic_ViT")

# ==========================================
# 3. THE DYNAMIC FACTORY
# ==========================================

def get_model(model_name, input_shape=(64, 64, 3), num_classes=200, pretrained=False, **kwargs):
    """
    FACTORY PATTERN: Takes the target model and automatically unpacks 
    all the specific YAML dictionary values (**kwargs) directly into the builder!
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
        raise ValueError(f"Model {model_name} not recognized!")

if __name__ == "__main__":
    # Smoke test to verify the dynamic architecture
    # Let's test building the "Micro-ViT" we defined for the laptop profile!
    test_kwargs = {
        "patch_size": 8,
        "projection_dim": 32,
        "num_heads": 4,
        "transformer_layers": 2,
        "dropout_rate": 0.2
    }
    
    print("Testing Dynamic ViT Build...")
    model = get_model("vit", **test_kwargs)
    model.summary()