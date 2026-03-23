# 🚀 Modern Computer Vision: SOTA Architectures & Enterprise MLOps

> Dataset: Tiny ImageNet (200 Classes, 64x64 Resolution)


## Table of contents

### 📖 Table of Contents (Section Tree)

1. Project Overview

2. Module 1 - Environment Setup & Data Engineering
- The Validation Data Bottleneck
- Hardcoded Infrastructure (YAML Configuration)
- The I/O Hardware Bottleneck (tf.data)

3. Module 2 - Architectural Design & The Factory Pattern
- The Factory Design Pattern
- The CNN Evolution (EfficientNetV2 & ConvNeXt)
- The Vision Transformer (ViT) Era

4. Module 3 - The Training Pipeline & MLOps
- Advanced Regularization (AdamW & Label Smoothing)
- Super-Convergence (OneCycle Learning Rate)
- Enterprise Experiment Tracking

5. Module 4 - Advanced Statistical Evaluation
- Core Classification Metrics
- Advanced Statistical Robustness (MCC & Top-20)
- Confidence Calibration (ECE)
- Enterprise Deployment Profiling (Latency & Size)

6. Quick Start & How to Run

## 📌 Project overview

This repository contains a high-performance, from-scratch implementation of an image classification pipeline targeting the **Tiny ImageNet** dataset. The goal of this project is to demonstrate modern Deep Learning engineering practices, transitioning from custom baseline architectures to State-of-the-Art (SOTA) models while maintaining a production-ready, hardware-agnostic codebase.

## Module 1 - ENviroment Setup & Data Engineering

### 📌 Overview

This module establishes the foundational infrastructure of the project. Before training advanced neural networks, we must ensure our data is mathematically aligned and our hardware is dynamically configured.

This module covers:
1. Solving the "Flat Directory" dataset problem.
2. Building a dynamic, hardware-agnostic configuration system.
3. Setting up the high-performance tf.data pipeline.


### ❓ Problem 1: The Validation Data Bottleneck

When working with the **Tiny ImageNet** dataset, the training data comes perfectly organized into 200 class-specific subfolders. However, the validation data is provided in a "flat" format—10,000 images dumped into a single `val/images/` directory.

Standard Deep Learning loaders (like TensorFlow's `image_dataset_from_directory` or PyTorch's `ImageFolder`) expect the folder name to represent the category label.

If we feed the raw validation folder to the model, the framework will assume all 10,000 images belong to a single class called "images," resulting in an immediate crash or a useless 0% accuracy metric.

🛠️ **The Solution: Automated Reorganization** (`data_loader.py`)

To solve this, we parse the provided `val_annotations.txt` file, which maps every image filename to its true WordNet ID (e.g., `val_0.jpg` → `n03444034`). The script automates the creation of the missing 200 subfolders and securely migrates the images into their correct categorical directories.


```python
# src/data_loader.py
import os
import shutil

# --- CONFIGURATION ---
# Target the raw validation directory
base_val_dir = 'data/tiny-imagenet-200/val'
images_dir = os.path.join(base_val_dir, 'images')
annotations_file = os.path.join(base_val_dir, 'val_annotations.txt')

def organize_validation_data():
    """
    Parses the ground-truth text file, generates missing class directories, 
    and migrates validation images to satisfy Keras categorical requirements.
    """
    # 1. Safety Check: Ensure we are operating in the correct root
    if not os.path.exists(annotations_file):
        print(f"ERROR: Could not find {annotations_file}. Are you in the project root?")
        return

    # 2. Parse the mapping file
    print("Step 5: Reading val_annotations.txt...")
    with open(annotations_file, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} validation images...")
    
    # 3. Iterative Migration
    for line in lines:
        parts = line.split('\t')
        image_name = parts[0]   # e.g., 'val_0.jpg'
        class_id = parts[1]     # e.g., 'n03444034'

        # 4. Directory Generation
        class_folder_path = os.path.join(images_dir, class_id)
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

        # 5. Secure File Move
        current_location = os.path.join(images_dir, image_name)
        new_location = os.path.join(class_folder_path, image_name)

        if os.path.exists(current_location):
            shutil.move(current_location, new_location)

    print("✅ SUCCESS: Validation folder is now organized by class folders!")

if __name__ == "__main__":
    organize_validation_data()
```

### ❓ Problem 2: Hardcoded Infrastructure

In academic tutorials, hyperparameters (Batch Size, Epochs, Learning Rate) are usually hardcoded directly into the training script. This is an anti-pattern in MLOps. If an engineer moves the code from a weak laptop to a powerful cloud GPU, they have to manually hunt down and change dozens of variables.


### 🛠️ The Solution: YAML Profile Injector (`utils.py`)

To make the codebase production-ready, we extract all variables into a `config.yaml` file. We then use a utility script to dynamically load specific "Hardware Profiles" (e.g., a "laptop" profile for quick debugging, or a "desktop" profile for full GPU training).

```python
# src/utils.py
import yaml
import os

def load_config(profile="laptop"):
    """
    Dynamically loads hyperparameter profiles from config.yaml and establishes
    absolute pathing for cross-platform execution.
    """
    # 1. Path Resolution: Find the config file relative to this specific script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    # 2. Profile Merging: Combine universal dataset rules with hardware-specific limits
    active_config = config['dataset']
    active_config.update(config[profile])

    # 3. State Tracking: Save the active profile name for MLOps logging
    active_config['profile_name'] = profile
    
    # 4. Automated File System Setup: Ensure save directories exist before training starts
    active_config['models_dir'] = os.path.join(base_dir, 'models')
    active_config['logs_dir'] = os.path.join(base_dir, 'logs')
    
    os.makedirs(active_config['models_dir'], exist_ok=True)
    os.makedirs(active_config['logs_dir'], exist_ok=True)
        
    return active_config
```


I defined specific "Hardware Profiles":

- Laptop Profile: CPU-only execution, small batch sizes, and 2-epoch limits for rapid code iteration and debugging.
- Desktop Profile: GPU-accelerated execution, maximized batch sizes, and high epochs for full training runs.

By changing a single string in `train.py` (`cfg = load_config(profile="laptop")`), the entire architecture, hardware allocation, and hyperparameter suite adapt instantly.

```YAML
# config.yaml
dataset:
  img_size: 64
  num_classes: 200
  train_dir: "data/tiny-imagenet-200/train"
  val_dir: "data/tiny-imagenet-200/val/images"

laptop:
  model_type: "vit"
  pretrained: false
  use_gpu: false
  batch_size: 16
  epochs: 1
  patience: 2
  learning_rate: 0.001
  max_lr: 0.003
  weight_decay: 1e-4       # AdamW Regularization
  label_smoothing: 0.1     # 10% uncertainty
  dropout_rate: 0.3        

desktop:
  model_type: "efficientnet"
  pretrained: true
  use_gpu: true
  batch_size: 128
  epochs: 100
  patience: 7
  learning_rate: 0.0005
  max_lr: 0.005
  weight_decay: 1e-4      
  label_smoothing: 0.1    
  dropout_rate: 0.4        # Higher dropout for deeper training
```

### ❓ Problem 3: The I/O Hardware Bottleneck

When training a deep neural network on 100,000 images, the GPU calculates gradients incredibly fast. However, if the GPU has to sit idle and wait for the CPU to fetch and decode the next batch of `.jpeg` files from the hard drive, training time doubles. This is known as an I/O Bottleneck.

**🛠️ The Solution: High-Performance `tf.data` Pipeline**

To keep hardware utilization near 100%, we implement a highly optimized `tf.data` pipeline using three critical operations:

1. `.cache()`: The dataset is read from the disk only during the very first epoch. The decoded images are then cached directly in RAM, reducing disk I/O to zero for all subsequent epochs.

2. **Strategic Augmentation**: The `data_augmentation` layer is mapped after the cache. If we mapped it before, we would cache the augmented images, meaning the model would see the exact same rotated/flipped images every epoch. By placing it after the cache, the CPU dynamically generates mathematically unique images on the fly.

3. `.prefetch(AUTOTUNE`): This enables **software pipelining**. While the GPU is processing Batch 1, the CPU is already preparing and augmenting Batch 2 in the background.

```python
# src/train.py (Dataset Pipeline Extract)
import tensorflow as tf
import keras

# --- ADVANCED DATA AUGMENTATION ---
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),  
    keras.layers.RandomZoom(0.1),      
], name="data_augmentation_pipeline")

def get_datasets(cfg):
    """Creates tf.data.Dataset objects with optimized memory and augmentation."""
    print("🚀 Loading Datasets...")
    
    # 1. Load raw data and One-Hot encode labels
    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg['train_dir'],
        label_mode='categorical',
        image_size=(cfg['img_size'], cfg['img_size']),
        batch_size=cfg['batch_size'],
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg['val_dir'],
        label_mode='categorical',
        image_size=(cfg['img_size'], cfg['img_size']),
        batch_size=cfg['batch_size'],
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    
    # 2. CACHE FIRST: Store decoded images in RAM to prevent disk I/O bottlenecks
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    # 3. AUGMENT SECOND: Dynamically augment cached images so they are unique every epoch
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # 4. PREFETCH LAST: Overlap CPU data preparation with GPU training execution
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
```


## Module 2 - Architectural Design & The Factory Pattern

###📌 Overview

In Phase 2, this project evolves from a static baseline script into a dynamic, production-ready Machine Learning pipeline. The focus shifts to implementing State-of-the-Art (SOTA) architectures and adopting industry-standard software design patterns.

This module covers:

1. The Factory Design Pattern (Modular Codebase).
2. The CNN Evolution (Baseline ResNet, EfficientNetV2, and ConvNeXt).
3. The Vision Transformer (ViT) Era and custom Tokenization.

### 🏗️ 1. Software Engineering: The Factory Pattern

In academic environments, training scripts are often hardcoded to a single model (e.g., `model = build_resnet()`). In a professional ML engineering environment, developers must test dozens of architectures rapidly without rewriting the training loop.

To solve this, we implement the **Factory Design Pattern**.

- **Decoupled Logic:** The training script (train.py) no longer imports specific models. Instead, it calls a single get_model() factory function.
- **Configuration Driven:** The factory reads the `model_type` string directly from `config.yaml` (e.g., `model_type: "convnext"`).
- **The Result:** To train a completely different architecture, we only need to change one word in a text file. The factory dynamically builds the correct model and applies the correct dropout rates automatically.

```python
# src/models.py (The Factory)
def get_model(model_name, input_shape=(64, 64, 3), num_classes=200, pretrained=False, dropout_rate=0.3):
    """
    FACTORY PATTERN: Returns the correct model architecture based on a string name.
    """
    if model_name == "resnet":
        return build_baseline_resnet(input_shape, num_classes)
    elif model_name == "efficientnet":
        return build_efficientnet(input_shape, num_classes, pretrained, dropout_rate)
    elif model_name == "convnext":
        return build_convnext(input_shape, num_classes, pretrained, dropout_rate)
    elif model_name == "vit": 
        # Custom ViT trains from scratch, ignoring 'pretrained' flag
        return build_vit(input_shape, num_classes, dropout_rate)
    else:
        raise ValueError(f"Model {model_name} not recognized!")
```

### 🧠 2. The Architectural Upgrades (CNNs)

The custom ResNet built in Phase 1 established a strong baseline, but modern Computer Vision relies on highly optimized architectures. Two cutting-edge Convolutional Neural Networks (CNNs) were added to the factory:

**A. EfficientNetV2 (The Efficiency King)**

Traditionally, engineers scaled CNNs by making them deeper or wider, leading to bloated, slow models. Google's EfficientNet solves this using **Compound Scaling**—using Neural Architecture Search to perfectly balance network depth, width, and image resolution simultaneously.

**B. ConvNeXt (The CNN Strikes Back)**

In 2020, Vision Transformers (ViTs) disrupted the field, outperforming traditional CNNs. In response, researchers at Meta (Facebook AI) modernized the standard ResNet using Transformer design philosophies, creating **ConvNeXt**.

- **Modernization:** It replaces standard 3x3 convolutions with massive 7x7 kernels, swaps ReLU for GELU activations, and replaces Batch Normalization with Layer Normalization.
- **The Result:** A pure CNN architecture that achieves higher accuracy than Vision Transformers while maintaining the training simplicity of a standard convolutional network.

```python
# src/models.py (State-of-the-Art CNNs)
def build_efficientnet(input_shape=(64, 64, 3), num_classes=200, pretrained=False, dropout_rate=0.3):
    """Builds a modern CNN using Keras Applications EfficientNetV2."""
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
    """Builds Meta's ConvNeXt (Tiny variant). The CNN that challenged Transformers."""
    inputs = keras.Input(shape=input_shape)
    weights_config = "imagenet" if pretrained else None
    
    # ConvNeXtTiny is highly parameter-efficient while maintaining SOTA accuracy
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
```

### 👁️ 3. The Vision Transformer (ViT) Era

Before ViTs, every image model used Convolutions. CNNs look at an image like you looking through a peephole—they scan the image pixel by pixel, learning edges, then shapes, then objects. Transformers, originally built for text, look at an entire sequence at once to understand global context.

However, standard pre-trained ViTs are designed for massive images (e.g., 224x224). When applied to Tiny ImageNet's 64x64 resolution, standard patch sizes (16x16) destroy the image geometry, resulting in only 16 total patches—far too few to learn meaningful patterns.

To solve this, we engineered a **Custom Mini-ViT** from scratch using TensorFlow subclassing.

**Step 1: Visual Tokenization (The Patch Creator)**

To make a Transformer read an image, we must translate the image into a "sentence." The custom `Patches` layer mathematically slices the 64x64 image into an 8x8 grid. This yields exactly 64 square patches, which are then flattened. The image is now a sentence containing 64 "words."

**Step 2: The Patch Encoder & `[CLS]` Token**

Transformers process all data simultaneously, meaning they possess no inherent concept of spatial order. Without intervention, the network wouldn't know if a patch belonged to the sky (top) or the grass (bottom).

- **Positional Embeddings:** The `PatchEncoder` learns a unique mathematical signature for each of the 64 positions and adds it to the patch data, restoring spatial awareness.

- **The Global** `[CLS]` **Token:** A learnable Classification token is concatenated to the beginning of the sequence. As it passes through the network, it interacts with every single patch, acting as an aggregator of global context.

**Step 3: Multi-Head Attention**

The ViT uses **Multi-Head Self-Attention**. In the very first layer, the patch representing a "dog's nose" can mathematically communicate with the patch representing the "dog's tail." By stacking Transformer blocks with Layer Normalization, GELU activations, and Residual connections, the architecture learns complex, global visual relationships immediately.

```python
# src/models.py (Custom Vision Transformer Implementation)
class Patches(layers.Layer):
    """Physically chops the image into a grid of squares and flattens them."""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # Extract sliding patches from the images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        # Flatten the squares into 1D vectors
        return tf.reshape(patches, [batch_size, -1, patch_dims])

class PatchEncoder(layers.Layer):
    """Projects patches and adds learned Positional Embeddings."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        # Input_dim must be num_patches + 1 to account for the CLS token
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, output_dim=projection_dim
        )
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, projection_dim),
            initializer="random_normal", trainable=True,
        )

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        patch_embeddings = self.projection(patch)
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        
        # Concatenate CLS token at the beginning of patch sequence
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        
        # Add positional embeddings to restore spatial awareness
        return patch_embeddings + self.position_embedding(positions)

def build_vit(input_shape=(64, 64, 3), num_classes=200, dropout_rate=0.3):
    """Builds the final ViT model optimized for 64x64 Tiny ImageNet images."""
    patch_size = 8  
    num_patches = (input_shape[0] // patch_size) ** 2 
    projection_dim = 64
    num_heads = 8
    transformer_layers = 4
    
    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # The Transformer Blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        encoded_patches = layers.Add()([x3, x2])
        
    # Final Classification via the [CLS] Token
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = representation[:, 0, :] # Extract only the CLS token
    representation = layers.Dropout(dropout_rate)(representation)
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    return keras.Model(inputs=inputs, outputs=outputs, name="Tiny_Custom_ViT")
```


## Module 3 - The Training Pipeline & MLOps

### 📌 Overview

Building a State-of-the-Art (SOTA) architecture is only half the battle. To extract maximum performance from these networks, the training loop must be heavily optimized.

This module covers:

1. The "Holy Trinity" of modern neural network training (AdamW, Label Smoothing, Data Augmentation).

2. Super-Convergence using the OneCycle Learning Rate Scheduler.

3. Automated MLOps and Enterprise Experiment Tracking.

### ⚙️ 1. The "Holy Trinity" of Modern Training

To push our models to peak accuracy on the challenging Tiny ImageNet dataset without overfitting, we implemented three advanced techniques standard in enterprise environments and Kaggle grandmaster solutions:

**A. Advanced Regularization (AdamW)**

When deep networks memorize training data instead of learning general patterns, we call it overfitting. Traditionally, engineers used L2 Regularization, which adds a penalty to the loss function to keep the model's mathematical weights small. However, standard Adam optimization interacts poorly with this penalty. We implemented **AdamW**, which mathematically decouples weight decay from the gradient update, directly shrinking the weights and drastically reducing overfitting.

**B. Confidence Calibration (Label Smoothing)**

Standard Categorical Crossentropy forces the model to be 100% confident (e.g., `[1.0, 0.0]`). On noisy datasets, this leads to extreme overconfidence and poor generalization. **Label Smoothing** alters the target distribution: 
$$y^{LS}=y^{hot}(1−\alpha)+ \frac{\alpha}{K}​$$

By adding a 10% margin of doubt (`label_smoothing=0.1`), the network is forced to learn robust, generalized features rather than memorizing exact pixel values.

**C. Dynamic Data Augmentation**

Built directly into the `tf.data` pipeline using Keras layers, this applies random horizontal flips, 10% rotations, and 10% zoom. Because Tiny ImageNet images are small (64×64), spatial augmentations are kept subtle to avoid destroying semantic meaning.

*Crucially, caching is executed before mapping the augmentations to ensure the model sees mathematically unique images every single epoch without I/O penalties.*

```python
# src/train.py (Compilation & Trinity Integration)
print("⚙️ Compiling with AdamW and Label Smoothing...")
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=cfg['learning_rate'],
        weight_decay=float(cfg['weight_decay']) 
    ),
    loss=keras.losses.CategoricalCrossentropy(
        label_smoothing=float(cfg['label_smoothing'])
    ),
    metrics=['accuracy']
)

# Advanced Data Augmentation Pipeline
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),  
    keras.layers.RandomZoom(0.1),      
], name="data_augmentation_pipeline")
```

### 📈 2. Advanced Callbacks & Super-Convergence

Static learning rates are a relic of the past. If the learning rate is too high, the model explodes; if it is too low, it gets stuck in bad local minima.

**The OneCycle Scheduler**

To solve this, we implemented a custom Keras 3 `Callback` for the **OneCycle Learning Rate Policy**.

- **Linear Warmup:** The learning rate starts low to prevent early gradient explosions, and linearly increases to a `max_lr` over the first 30% of training. This acts as an exploratory phase, allowing the model to safely traverse steep loss landscapes. The high peak learning rate acts as a form of "super-regularization," violently bouncing the model out of sharp, bad local minima.

- **Cosine Decay:** For the remaining 70% of training, the learning rate smoothly decays following a cosine curve, allowing the model to precisely settle into the optimal global minimum.

**Early Stopping & Resource Management**

Cloud computing is expensive. If a model begins to overfit, continuing to train it wastes money and degrades performance.

- Our Keras `EarlyStopping` callback actively monitors `val_accuracy`.

- If the model fails to improve for a set number of epochs (defined dynamically by the YAML `patience` variable), training halts automatically.

- `restore_best_weights=True` ensures that we don't save an overfitted model, reverting the architecture to the exact moment it achieved peak performance.

```python
# src/train.py (The Custom OneCycle Scheduler)
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        pct = self.step / self.total_steps
        
        # 30% Linear Warmup
        if pct < 0.3:
            lr = self.max_lr * (pct / 0.3)
        # 70% Cosine Decay
        else:
            progress = (pct - 0.3) / 0.7
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.max_lr * cosine_decay
            
        lr = max(lr, 1e-6)
        self.model.optimizer.learning_rate = float(lr)
        self.step += 1
```

### 📊 3. MLOps: Enterprise Experiment Tracking

A hallmark of production-level engineering is reproducibility. Manually writing down terminal outputs is error-prone and unscalable. To treat this project like a true enterprise deployment, we built custom **Experiment Tracker Callbacks**.

1. The `DynamicHistoryLogger`: Saves the full decimal values of every metric (`Loss`, `Accuracy`, `Val_Loss`, `Val_Accuracy`) at the end of every single epoch into a dedicated CSV file. This allows for high-fidelity post-training visualization.

2. The `ExperimentTracker`: Acts as a master database. When a training run concludes, this callback calculates the total elapsed time, extracts the peak metrics, and records them alongside the exact hyperparameters used (`Learning_Rate`, `Dropout_Rate`, `Batch_Size`, etc.) into a running `experiment_logs.csv`.

```python
# src/train.py (MLOps Master Tracker Extract)
class ExperimentTracker(keras.callbacks.Callback):
    def __init__(self, cfg, log_filename="experiment_logs.csv"):
        super().__init__()
        self.cfg = cfg
        self.log_file = os.path.join(self.cfg['logs_dir'], log_filename)
        self.best_val_acc = 0.0
        self.best_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_run += 1
        # Track the best epoch based on validation accuracy
        current_val_acc = logs.get('val_accuracy', 0)
        if current_val_acc >= self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_metrics = logs.copy()

    def on_train_end(self, logs=None):
        # Calculate training duration
        self.end_time = datetime.datetime.now()
        duration = self.end_time - self.start_time
        
        # Assemble the automated data row
        row = {
            "Profile": self.cfg.get('profile_name', 'unknown').upper(),
            "Date": self.start_time.strftime("%d/%m/%Y"),
            "Duration": f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m",
            "Model": self.cfg['model_type'].upper(),
            "Pretrained": self.cfg.get('pretrained', False),
            "Epochs_Run": self.epochs_run,
            "Train_Accuracy": round(self.best_metrics.get('accuracy', 0), 4),
            "Val_Accuracy": round(self.best_metrics.get('val_accuracy', 0), 4),
            "Dropout_Rate": self.cfg.get('dropout_rate', 0.3),
            "Learning_Rate": self.cfg.get('learning_rate', 0.001),
        }

        # Securely append to the master CSV database
        fieldnames = list(row.keys())
        file_exists = os.path.isfile(self.log_file)
        
        with open(self.log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() 
            writer.writerow(row)
```

## Module 4 - Advanced Statistical Evaluation & Deployment Profiling

### 📌 Overview

In a multi-class problem with 200 categories, a single "Accuracy" metric is dangerously misleading. A model could be perfectly predicting "Cars" while failing entirely at "Animals," or it might be making highly confident, yet completely incorrect, predictions.

To prove the robustness and production-readiness of our architectures, we implemented a custom, multi-faceted statistical evaluation suite that runs entirely post-training.

This module covers:

1. Core Classification & High-Confidence Metrics.

2. Advanced Statistical Robustness (Top-20, MCC).

3. Confidence & Reliability Calibration (ECE, AUC-ROC).

4. Hardware & Deployment Profiling (Latency, Model Size).


### 📊 1. Core Classification & The "Lucky Guess" Problem

Standard Top-1 Accuracy treats a 15% confident guess and a 99% confident guess identically. To filter out "lucky guesses," we engineered a **High-Confidence Accuracy (>90%)** metric. This forces the model to put its money where its mouth is—only predictions where the Softmax probability exceeds 0.90 are counted as correct.

We also extract the **Macro F1-Score**. By calculating the harmonic mean of Precision and Recall across all 200 classes evenly, we prevent majority classes from hiding the poor performance of minority classes.

```python
# src/evaluation (Core Metrics Extract)
# Standard Top-1 Accuracy
standard_acc = np.mean(y_true_classes == y_pred_classes)

# High-Confidence Accuracy (>90%)
max_probs = np.max(y_pred_probs, axis=1)
correct_and_confident = np.sum((y_pred_classes == y_true_classes) & (max_probs >= 0.90))
high_conf_acc = correct_and_confident / len(y_true_classes)

# Macro F1-Score
macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
```

### 🛡️ 2. Advanced Statistical Robustness

Asking a network to get the exact right answer on the first try out of 200 possible classes is a brutal standard.

- **Top-20 Accuracy:** We implemented a highly optimized, vectorized check to see if the true label exists within the top 10% (Top 20) of the model's highest-confidence predictions. This tests the network's general "intuition."

- **Matthews Correlation Coefficient (MCC):** Widely considered the most mathematically reliable statistical rate for multi-class evaluation. It only yields a high score if the network achieved good results across all four confusion matrix quadrants (True Positives, False Negatives, etc.).

```python
# src/evaluation (Vectorized Top-20 & MCC)
# 1. Top-20 Accuracy (Vectorized for maximum speed)
# Gets the indices of the top 20 probabilities for every single image
top20_preds = np.argsort(y_pred_probs, axis=1)[:, -20:]
# Checks if the true class is anywhere inside those top 20 predictions
top20_acc = np.mean(np.any(top20_preds == y_true_classes[:, None], axis=1))

# 2. Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
```

### 🔎 3. Confidence & Reliability Calibration

A model that boasts 99% accuracy but is extremely overconfident when making mistakes is dangerous to deploy in the real world (e.g., an autonomous vehicle being "99.9% sure" a pedestrian is a shadow).

To measure this, we wrote a custom calculator for **Expected Calibration Error (ECE)**. If a model outputs a 90% probability, it should genuinely be correct exactly 90% of the time. ECE calculates the gap between the model's reported confidence and its actual accuracy across 10 probability bins. **(Lower ECE is better)**.

```python
# src/evaluation (Expected Calibration Error Calculator)
confidences = np.max(y_pred_probs, axis=1)
predictions = np.argmax(y_pred_probs, axis=1)
accuracies = (predictions == y_true_classes)

# Split into 10 confidence bins (0.0 to 1.0)
num_bins = 10
bins = np.linspace(0.0, 1.0, num_bins + 1)
bin_indices = np.digitize(confidences, bins, right=True)

ece = 0.0
for b in range(1, num_bins + 1):
    mask = bin_indices == b
    if np.any(mask):
        bin_accuracy = np.mean(accuracies[mask])
        bin_confidence = np.mean(confidences[mask])
        bin_weight = np.sum(mask) / len(confidences)
        # Calculate the absolute gap between confidence and accuracy
        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
```

### ⚙️ 4. Enterprise Deployment Profiling

This final phase separates Data Scientists from Machine Learning Engineers. In a production environment, an architecture is restricted by the hardware it runs on (e.g., mobile phones, edge devices, cloud servers).

We built an automated physical profiling script that loads each .keras file and stresses the hardware to extract three vital deployment metrics:

1. **Model Size (MB):** The physical disk space required to store the weights.

2. **Parameter Count:** The total number of mathematical weights. This dictates the RAM/VRAM footprint required just to load the model into memory.

3. **Inference Latency (ms/image):** How fast the model can process a single image. We implement a "warm-up" pass first (to build the computation graph), followed by a timed 10-batch execution loop to calculate the precise millisecond latency.

```python
# src/evaluation (Hardware Profiling Extract)
# 1. Model Size on Disk (MB)
file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

# 2. Parameter Count (Millions)
total_params = model.count_params()

# 3. Inference Latency (ms per image)
# WARM-UP: Neural networks are slow on the first pass. Run once and ignore.
_ = model.predict(sample_images, verbose=0)

# TIMED RUN: Run inference 10 times to get a stable average
num_runs = 10
start_time = time.time()
for _ in range(num_runs):
    _ = model.predict(sample_images, verbose=0)
end_time = time.time()

# Calculate precise milliseconds per image
avg_time_per_image_ms = ((end_time - start_time) / (num_runs * 64)) * 1000
```

## 🚀 Quick Start

Follow these steps to replicate the environment, download the dataset, and train the models from scratch.

1. **Environment Setup:** 

To ensure perfect reproducibility and avoid dependency conflicts, we use Miniconda to manage our Python environment.

- If you do not have Miniconda installed, download it from the [Official Conda Website](https://docs.conda.io/en/latest/miniconda.html).

Open your terminal and run the following commands to create and activate a new virtual environment:

```Bash
# Create a new Conda environment with Python 3.10
conda create -n vision_portfolio python=3.10 -y

# Activate the environment
conda activate vision_portfolio
```

2. Install Dependencies

With your environment activated, install all the required Deep Learning and data visualization libraries:

```Bash
# Install requirements via pip
pip install -r requirements.txt
```

3. Download the Dataset

The Tiny ImageNet dataset is hosted by Stanford University. You need to download it and place it in the `data/` directory.

```Bash
# Create the data directory
mkdir data
cd data

# Download the dataset directly from Stanford
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Unzip the dataset
unzip tiny-imagenet-200.zip
cd ..
```

4. Fix the Validation Data Bottleneck

As detailed in Module 1, the raw validation dataset is entirely flat and incompatible with modern deep learning loaders. Run the data loader script to automatically organize the 10,000 validation images into their correct 200 categorical subfolders:

```Bash
python -m src.data_loader
```

Wait for the console to print: `✅ SUCCESS: Validation folder is now organized by class folders!`

5. Configure Your Hardware Profile

Open `config.yaml` in the root directory. This file dictates which model to build and what hardware limits to apply.
Next, open `src/train.py` and adjust the configuration loader at the very top of the script to match your current machine:

```Python
# Inside src/train.py
# Use "laptop" for quick CPU testing, or "desktop" for full GPU training
cfg = load_config(profile="desktop")
```

6. Train the Network

Execute the main training loop. The Factory Pattern will automatically build your selected model, initiate the `tf.data` pipeline, and begin tracking metrics.

```Bash
python -m src.train
```

7. Evaluate the Results

Once training is complete (or triggered by Early Stopping), open the Jupyter Notebook inside the `notebooks/` directory to generate the Top-20 Accuracy, Confusion Matrix, and Hardware Footprint dashboards.

```Bash
jupyter notebook notebooks/Model_Evaluation_and_Dashboards.ipynb
```


## 📂 Folder Structure

<pre>
📦 TinyImageNet_Project
├── 📁 <a href="https://github.com/victorperone/TinyImageNet_Project/tree/main/data">data</a> # (Git-ignored)
│   ├── 📁 tiny-imagenet-200
│   └── 🗄️ tiny-imagenet-200.zip
├── 📁 <a href="https://github.com/victorperone/TinyImageNet_Project/tree/main/logs">logs</a>
│   ├── 📄 experiment_logs.csv
│   └── 📄 efficientnet_history_09-03-2026.csv
├── 📁 <a href="https://github.com/victorperone/TinyImageNet_Project/tree/main/models">models</a> # (Git-ignored)
│   ├── 🧠🤖 baseline_resnet_best.keras
│   ├── 🧠🤖 efficientnet_best.keras
│   └── 🧠🤖 vit_best.keras
├── 📁 <a href="https://github.com/victorperone/TinyImageNet_Project/tree/main/notebooks">notebooks</a>
│   └── 📓 Model_Evaluation_and_Dashboards.ipynb
├── 📁 <a href="https://github.com/victorperone/TinyImageNet_Project/tree/main/src">src</a>
│   ├── 🐍 init.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 models.py
│   ├── 🐍 train.py
│   └── 🐍 utils.py
├── ⚙️ config.yaml
├── 📄 requirements.txt
└── 📄 README.md
</pre>









# 4. Missing modern tricks

## You’re leaving accuracy on the table without:
## MixUp / CutMix
## RandAugment / AutoAugment
## EMA (Exponential Moving Average)
## Cosine LR scheduling (instead of static or simple max_lr)

























































































































# Step 5: Validation Data Reorganization

## 📌 Overview
This step handles the critical pre-processing of the Tiny ImageNet validation dataset. While the training data comes pre-organized into class-specific subfolders, the validation data is provided in a "flat" format that is incompatible with standard deep learning data loaders.

## ❓ The Problem
If you look inside `data/tiny-imagenet-200/val/images`, you will see **10,000 images** all sitting in a single directory. 
* **Standard Loaders:** Functions like `image_dataset_from_directory` (TensorFlow) or `ImageFolder` (PyTorch) expect each folder name to represent the category label.
* **The Conflict:** Without reorganization, these libraries will treat all 10,000 validation images as belonging to a single class, leading to a 0% validation accuracy during training.

## 🛠 The Solution: `data_loader.py`
The provided script automates the reorganization by using the ground-truth mappings found in `val_annotations.txt`.

### How the Script Works:
1. **Parses Annotations:** It reads the `val_annotations.txt` file, which maps every image filename (e.g., `val_0.jpg`) to its corresponding class ID (e.g., `n03444034`).
2. **Creates Directories:** It checks if a folder for that specific class ID exists within the `val/images` directory; if not, it creates it.
3. **Migrates Files:** It physically moves each image from the root validation folder into its new, class-specific subfolder.

### Script Implementation:
```python
import os
import shutil

# Paths configuration
val_dir = 'data/tiny-imagenet-200/val'
images_dir = os.path.join(val_dir, 'images')
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

def fix_validation_set():
    # Verify annotation file exists
    if not os.path.exists(annotations_file):
        print("Error: Annotations file not found.")
        return

    # Read the mapping file
    with open(annotations_file, 'r') as f:
        lines = f.readlines()

    # Move images to class folders
    for line in lines:
        parts = line.split('\t')
        img_name = parts[0]
        class_id = parts[1]

        target_folder = os.path.join(images_dir, class_id)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        src = os.path.join(images_dir, img_name)
        dst = os.path.join(target_folder, img_name)
        
        if os.path.exists(src):
            shutil.move(src, dst)

if __name__ == "__main__":
    fix_validation_set()
    print("Validation data reorganized successfully.")
```

How to Verify it Worked

Don't just trust the message! Let's check the folders:
- In the VS Code sidebar, expand data > tiny-imagenet-200 > val > images.
- Before, you saw a long list of .jpg files.
- Now, you should see a long list of folders (like n01443537, n01629819).
- Click into one of those folders—you should see the images inside.

val/
└── images/
    ├── n01443537/
    │   └── val_0.jpg
    ├── n01629819/
    │   └── val_15.jpg
    └── ... (200 total folders)










# Phase 1: High-Performance Data Pipeline

## 🛠 Technical Implementation
To handle the 100,000 training images of Tiny ImageNet, I implemented a data pipeline using the `tf.data` API. This approach is superior to simple memory loading as it prevents OOM (Out of Memory) errors and optimizes hardware utilization.

### Key Optimization Techniques:

1. **Categorical Labeling:**
   - With 200 distinct classes, the labels are converted to **One-Hot Encoded vectors**. This allows the model to calculate probability distributions across all potential categories using a Softmax output layer.

2. **Memory Caching (`.cache()`):**
   - By caching the dataset, the images are only read from the disk during the first epoch. In subsequent epochs, the data is pulled directly from RAM, reducing I/O bottlenecks by up to 80%.

3. **Software Pipelining (`.prefetch()`):**
   - I utilized `AUTOTUNE` to enable **overlapping**. This ensures that the CPU prepares the next batch of images while the model is still processing the current batch. This keeps the processing unit (GPU/CPU) at near 100% utilization.

4. **Reproducibility:**
   - A fixed random seed (42) was used during the training split to ensure that experiments are reproducible and that performance gains are due to architecture changes, not random data ordering.

## 📊 Dataset Statistics
- **Input Resolution:** 64 x 64 pixels (RGB)
- **Total Training Samples:** 100,000
- **Total Validation Samples:** 10,000
- **Total Classes:** 200








# 🚀 Modern Computer Vision: Tiny ImageNet Classification

## 📌 Project Overview
This repository contains a high-performance, from-scratch implementation of an image classification pipeline targeting the **Tiny ImageNet** dataset (200 classes, 64x64 resolution). 

The goal of this project is to demonstrate modern Deep Learning engineering practices, transitioning from custom baseline architectures to State-of-the-Art (SOTA) models while maintaining a production-ready, hardware-agnostic codebase.

---

## 🏗️ Phase 1: The Foundation & Baseline
In Phase 1, the infrastructure was established to handle large-scale data efficiently and establish a solid accuracy baseline.

### Key Engineering Features:
1. **High-Performance Data Pipeline (`tf.data`):**
   - Implemented `cache()` and `prefetch(AUTOTUNE)` to prevent CPU/GPU bottlenecking.
   - Handled dataset reorganization to convert "flat" validation data into framework-compliant subdirectories.

2. **Custom Residual Architecture (ResNet):**
   - Built a lightweight ResNet from scratch.
   - Utilized skip connections to prevent the vanishing gradient problem, optimizing the network specifically for the 64x64 spatial constraints of Tiny ImageNet.

3. **Dynamic Hardware Configuration (`config.yaml`):**
   - Implemented a dual-profile configuration system.
   - **Laptop Profile:** CPU-only execution with small batch sizes for rapid code iteration and debugging.
   - **Desktop Profile:** GPU-accelerated execution (RTX series) with maximized batch sizes and extended epochs for full training runs.

4. **Modern Optimization (OneCycle Policy):**
   - Replaced static learning rate decay with a custom Keras 3 `Callback` implementing the **OneCycle Learning Rate Scheduler**. 
   - Uses linear warmup to a maximum learning rate followed by cosine decay, allowing for faster convergence and super-convergence dynamics.

---

## 🚀 Phase 2: Modern Architectures & Production Patterns
In Phase 2, the project evolved from a static baseline script into a dynamic, production-ready Machine Learning pipeline. The focus shifted to implementing state-of-the-art (SOTA) architectures and adopting industry-standard software design patterns.

### 1. The Factory Design Pattern (Modular Codebase)
In academic tutorials, training scripts are often hardcoded to a single model. In a professional environment, engineers need to test dozens of models rapidly without rewriting the training loop. 
To solve this, I implemented the **Factory Design Pattern**:
* **How it works:** Instead of importing a specific model (e.g., `build_resnet()`), the training script calls a single `get_model()` function.
* **The YAML Link:** The factory reads the `model_type` string directly from the `config.yaml` file (e.g., `model_type: "efficientnet"`).
* **The Result:** The codebase is now completely decoupled. To train a completely different architecture, I only need to change one word in a text file, and the factory automatically builds the correct model and names the saved weight files dynamically to prevent accidental overwrites.

### 2. EfficientNetV2 (State-of-the-Art CNN)
The custom ResNet built in Phase 1 was a great baseline, but modern Computer Vision relies on highly optimized architectures. I implemented **EfficientNetV2** (developed by Google).
* **The Problem with Old CNNs:** Traditionally, to make a model more accurate, engineers would just make it deeper (adding more layers) or wider (adding more filters). This makes models extremely slow and massive.
* **The EfficientNet Solution:** It uses "Compound Scaling." Instead of scaling just one dimension, it uses Neural Architecture Search (AI designing AI) to perfectly balance **depth**, **width**, and **image resolution** simultaneously.
* **Why V2?:** EfficientNetV2 introduces Fused-MBConv blocks, which drastically improve training speed and parameter efficiency compared to the original version.



### 3. Transfer Learning (Pre-trained Weights)
Training a massive model from scratch on 100,000 images requires immense computational power and time. To optimize resource usage, I integrated configurable **Transfer Learning**.
* **The Concept:** Instead of initializing the network with random, "dumb" weights, we load a version of EfficientNet that has already spent thousands of GPU hours learning to identify shapes, textures, and objects on the **ImageNet** dataset (1.2 million images, 1000 classes).
* **Fine-Tuning:** By setting `pretrained: true` in the YAML config, the model imports these "smart" weights. We discard the original 1000-class output layer, attach a fresh 200-class layer for our Tiny ImageNet task, and train it. 
* **The Impact:** The model converges to a high accuracy in a fraction of the time, effectively standing on the shoulders of Google's massive compute clusters.



### 4. Early Stopping (Compute Resource Management)
When training deep neural networks, there is a constant danger of **Overfitting**—the point where the model stops learning general patterns and starts memorizing the specific training images. Once this happens, its accuracy on new, unseen data gets worse.
* **The Implementation:** I added a Keras `EarlyStopping` callback that actively monitors the `val_accuracy` (Validation Accuracy) at the end of every epoch.
* **Patience:** I configured a `patience` parameter. If the model fails to improve for a set number of epochs (e.g., 7 epochs on the desktop profile), the callback halts the training process entirely.
* **Restoring Best Weights:** To ensure we don't save an overfitted model, the callback is configured with `restore_best_weights=True`. It automatically discards the bad epochs and reverts the model to the exact moment it achieved its highest accuracy. This prevents wasted cloud compute costs and guarantees optimal model deployment.


## 📂 Project Structure
```text
TinyImageNet_Project/
├── config.yaml          # Hardware and training hyperparameter profiles
├── data/                # Dataset (Git-ignored)
├── models/              # Saved model weights (.keras)
├── notebooks/           # EDA, Saliency Maps, and Confusion Matrices
├── src/
│   ├── __init__.py      # Package indicator
│   ├── data_loader.py   # Validation data reorganization script
│   ├── models.py        # Neural Network architectures
│   ├── train.py         # Main training loop and callbacks
│   └── utils.py         # YAML parsers and helpers
└── README.md            # Project documentation
```

## ⚙️ Quick Start

1. **Environment Setup:**
Ensure you have a Conda environment with Python 3.10 and Keras 3 installed.

2. **Download Data:**
Place the dataset in /data/tiny-imagenet-200/.

3. **Fix Validation Data:**
Run python -m src.data_loader to properly format the validation directory.

4. **Train:**
Select your profile in train.py (cfg = load_config("laptop")) and run:

```bash
python -m src.train
```

## 📅 Roadmap

    [x] Phase 1: tf.data Pipeline, Custom ResNet Baseline, OneCycle Policy.

    [ ] Phase 2: EfficientNetV2 & ConvNeXt implementations.

    [ ] Phase 3: Vision Transformers (ViTs) and Attention Maps.

    [ ] Phase 4: Advanced Augmentation (Mixup/CutMix).

    [ ] Phase 5: Knowledge Distillation.











The Result

You now have:
1. A dynamic learning rate (OneCycle) pushing the model out of bad local minima.
2. A dynamic optimizer (AdamW) keeping the weights strictly regularized.
3. A calibrated loss function (Label Smoothing) preventing overconfidence.
4. Infinite data generation (Data Augmentation) preventing memorization.








# The Story of ConvNeXt (The CNN Strikes Back)

In 2020, Vision Transformers (ViTs) hit the scene and absolutely crushed traditional CNNs like ResNet. Everyone thought CNNs were dead.

But in 2022, engineers at Meta (Facebook AI) asked a brilliant question: "What if we took a standard ResNet and modernized it using all the tricks we learned from Transformers?"

They changed the activation functions from ReLU to GELU, swapped BatchNorm for LayerNorm, increased the kernel sizes from 3x3 to 7x7, and changed how the network bottlenecks worked. The result was ConvNeXt—a pure CNN that trained faster and actually beat Vision Transformers!





## 🚀 Phase 2: State-of-the-Art Architectures & MLOps
In Phase 2, this project evolved from a static baseline into a dynamic, production-ready Machine Learning pipeline. The focus shifted to implementing State-of-the-Art (SOTA) architectures, adopting industry-standard software design patterns, and establishing an MLOps experiment tracking system.

### 🏗️ 1. Software Engineering: The Factory Pattern
In academic environments, training scripts are often hardcoded to a single model. In a professional ML engineering environment, developers must test dozens of architectures rapidly without rewriting the training loop. 

To solve this, I implemented the **Factory Design Pattern**:
* **Decoupled Logic:** The training script (`train.py`) no longer imports specific models. Instead, it calls a single `get_model()` factory function.
* **Configuration Driven:** The factory reads the `model_type` string directly from `config.yaml` (e.g., `model_type: "convnext"`).
* **The Result:** To train a completely different architecture, I only need to change one word in a text file. The factory dynamically builds the correct model, applies the correct dropout rates, and names the saved weight files automatically to prevent accidental overwrites.

### 🧠 2. The Architectural Upgrades
The custom ResNet built in Phase 1 established a baseline, but modern Computer Vision relies on highly optimized architectures. Two cutting-edge models were added to the factory:

#### A. EfficientNetV2 (The Efficiency King)
Traditionally, engineers scaled CNNs by making them deeper or wider, leading to bloated, slow models. Google's EfficientNet solves this using **Compound Scaling**—using Neural Architecture Search to perfectly balance network depth, width, and image resolution simultaneously. 
* **V2 Upgrades:** EfficientNetV2 introduces Fused-MBConv blocks, replacing standard depthwise convolutions in early layers, which drastically improves training speed and parameter efficiency.


#### B. ConvNeXt (The CNN Strikes Back)
In 2020, Vision Transformers (ViTs) disrupted the field, outperforming traditional CNNs. In response, researchers at Meta (Facebook AI) modernized the standard ResNet using Transformer design philosophies, creating **ConvNeXt**.
* **Modernization:** It replaces standard $3 \times 3$ convolutions with massive $7 \times 7$ kernels, swaps ReLU for GELU activations, and replaces Batch Normalization with Layer Normalization. 
* **The Result:** A pure CNN architecture that achieves higher accuracy than Vision Transformers while maintaining the training simplicity of a standard convolutional network.


### ⚙️ 3. The "Holy Trinity" of Modern Training
To extract maximum performance from these networks, the training loop was upgraded with three advanced techniques used in enterprise environments and Kaggle competitions:

1. **Advanced Regularization (AdamW):** Standard Adam optimization handles weight decay (L2 regularization) poorly. I implemented **AdamW**, which mathematically decouples weight decay from the gradient update. This tightly controls the weights, drastically reducing overfitting on deep networks.
2. **Confidence Calibration (Label Smoothing):** Standard Categorical Crossentropy forces the model to be 100% confident (e.g., `[1.0, 0.0]`), leading to extreme overconfidence. **Label Smoothing** alters the target distribution: $y^{LS} = y^{hot}(1 - \alpha) + \frac{\alpha}{K}$. By adding a 10% margin of doubt, the network is forced to learn better, more generalized features rather than memorizing exact pixel values.
3. **Dynamic Data Augmentation:** Built directly into the `tf.data` pipeline using Keras layers, this applies random horizontal flips, 10% rotations, and 10% zoom. Because Tiny ImageNet images are small ($64 \times 64$), spatial augmentations are kept subtle to avoid destroying semantic meaning. Caching is strategically executed *before* mapping the augmentations to ensure the model sees mathematically unique images every single epoch.

### 🔄 4. Transfer Learning Integration
Training deep networks from scratch on 100,000 images is computationally expensive. The pipeline now supports configurable **Transfer Learning**.
* By setting `pretrained: true` in the YAML config, the factory imports weights trained on the massive ImageNet-1k dataset. 
* The original 1000-class output layer is discarded and replaced with a newly initialized 200-class layer tailored for Tiny ImageNet. This allows the model to leverage pre-learned edge and texture detection, achieving convergence in a fraction of the time.

### 📊 5. MLOps & Experiment Tracking
To treat this project like a true enterprise deployment, I built a custom **Experiment Tracker Callback**.
* **Automated Logging:** The moment a training run finishes (or triggers Early Stopping), the callback calculates total elapsed time and extracts the peak metrics (Train/Val Loss and Accuracy).
* **CSV Database:** It automatically appends this data, along with the specific hyperparameters used (`Learning_Rate`, `Dropout_Rate`, `Batch_Size`, etc.), to a running `experiment_logs.csv` database. 
* **Impact:** This guarantees a permanent, spreadsheet-ready record of all hyperparameter tuning, eliminating the need to manually track terminal outputs.







# Phase 3 Explainig ViT

This is where we leave traditional Computer Vision behind and enter the modern era of Artificial Intelligence.

Vision Transformers (ViTs) completely changed the world in 2020. Before ViTs, every image model used Convolutions (CNNs). Convolutions look at an image like you looking through a peephole—they scan the image pixel by pixel, learning edges, then shapes, then objects.

Transformers don't do that. Transformers were originally built for text (like ChatGPT). They look at an entire sentence at once to understand the context.

To make a Transformer understand an image, we have to trick it into thinking the image is a sentence!
The Theory: How ViTs Work

Here are the 4 steps of a Vision Transformer, translated from theory to practice:
1. Patches (The "Words"): We take our 64×64 Tiny ImageNet image and chop it into a grid of smaller squares (e.g., 8×8 pixels). Each square is treated like a "word" in a sentence. Since 64÷8=8, we get an 8×8 grid, resulting in 64 patches.
2. Linear Projection (Flattening): A neural network can't read a 2D square. We flatten each 8×8 patch into a single 1D line of numbers.
3. Positional Embedding (The "Order"): Because Transformers look at everything at the exact same time, they don't know if a patch came from the top-left (the sky) or the bottom-right (the grass). We add a mathematical "tag" to each patch saying, "I am patch #1, I am patch #2", etc.
4. Multi-Head Attention (The "Magic"): The network compares every patch to every other patch simultaneously. The "sky" patch pays attention to the "bird" patch, and the "dog" patch pays attention to the "grass" patch. This global context is why ViTs are so powerful.


Input Image (64x64x3)
        │
        ▼
 ┌───────────────────┐
 │   Patch Creator   │
 │ 8x8 patches       │
 │ → 64 patches      │
 └───────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ Flatten Each Patch       │
 │ 8×8×3 = 192 values       │
 │ → vector size 192        │
 └──────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ Patch Projection         │
 │ Dense: 192 → 64          │
 │ (Patch Embedding)        │
 └──────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ Add Positional Embedding │
 │ tells model where patch  │
 │ is located               │
 └──────────────────────────┘
        │
        ▼
  ┌─────────────────────────┐
  │  Transformer Block ×4   │
  │                         │
  │ LayerNorm               │
  │ MultiHead Attention     │
  │ Residual Connection     │
  │ LayerNorm               │
  │ MLP (Dense + GELU)      │
  │ Residual Connection     │
  └─────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ Flatten All Patches      │
 │ 64 × 64 = 4096 features  │
 └──────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ Dropout                  │
 └──────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ Dense Softmax (200)      │
 │ Tiny ImageNet Classes    │
 └──────────────────────────┘


| Step        | Shape            |
| ----------- | ---------------- |
| Input       | `(B, 64, 64, 3)` |
| Patches     | `(B, 64, 192)`   |
| Projection  | `(B, 64, 64)`    |
| Transformer | `(B, 64, 64)`    |
| Flatten     | `(B, 4096)`      |
| Classifier  | `(B, 200)`       |



## 👁️ Phase 3: The Vision Transformer (ViT) Era
In 2020, Vision Transformers completely disrupted Computer Vision, proving that architectures originally designed for Natural Language Processing (like ChatGPT) could outperform traditional CNNs on image tasks.

However, standard pre-trained ViTs are designed for large images (e.g., $224 \times 224$). When applied to Tiny ImageNet's $64 \times 64$ resolution, standard patch sizes ($16 \times 16$) destroy the image geometry, resulting in only 16 total patches—far too few for the attention mechanisms to learn meaningful patterns. 

To solve this, I engineered a **Custom Mini-ViT** from scratch using TensorFlow/Keras subclassed layers, specifically optimized for $64 \times 64$ images.

### 🧩 1. The Patch Creator (Visual Tokenization)
Transformers process sequences of words (tokens). To make a Transformer read an image, we must translate the image into a sequence.
* I built a custom `Patches` layer that mathematically slices the $64 \times 64$ image into an $8 \times 8$ grid. 
* This yields exactly 64 square patches. Each patch is then flattened from a 2D matrix into a 1D vector. In the eyes of the Transformer, the image is now a "sentence" containing 64 "words."

### 📍 2. The Patch Encoder & The `[CLS]` Token
Unlike CNNs, Transformers process all data simultaneously, meaning they possess no inherent concept of spatial order. Without intervention, the network wouldn't know if a patch belonged to the sky (top) or the grass (bottom).
* **Positional Embeddings:** I implemented a custom `PatchEncoder` that learns a unique mathematical signature for each of the 64 positions and adds it to the patch data, restoring spatial awareness.
* **The Global `[CLS]` Token:** Following the original Google Research paper, I concatenated a learnable Classification (`[CLS]`) token to the very beginning of the sequence (making it 65 tokens long). As this token passes through the network, it interacts with every single image patch, acting as an aggregator of global context.



### 🧠 3. Multi-Head Attention (Global Context)
Traditional CNNs suffer from a restricted "Receptive Field"—they look at images through a tiny moving window (e.g., $3 \times 3$ pixels) and only understand the whole image after dozens of layers.
* The ViT uses **Multi-Head Self-Attention**. In the very first layer, the patch representing a "dog's nose" can mathematically communicate with the patch representing the "dog's tail" on the other side of the image. 
* By stacking 4 Transformer blocks with Layer Normalization, GELU activations, and Residual (Skip) connections, the architecture learns complex, global visual relationships immediately.
* **Classification:** At the end of the network, we discard the 64 image patches and pass *only* the aggregated `[CLS]` token into a dense Softmax layer to generate the final 200-class prediction.











