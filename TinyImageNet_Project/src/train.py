import sys
import os
import math
import datetime
import csv

# 1. Take the "blinders" off: Add the project root to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Load Config FIRST so we can set hardware profiles BEFORE TensorFlow boots up
from src.utils import load_config
# Change this to "desktop" when you get to the RTX 3060
cfg = load_config(profile="laptop") 

if not cfg['use_gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] LAPTOP MODE ACTIVE")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE ACTIVE")

# 3. Import everything else NOW
import tensorflow as tf
import keras
from keras import mixed_precision
from src.models import get_model
from src.plotting import generate_training_plot
from src.evaluation import evaluate_model

# ------------------------------------------------------------
# Enable mixed precision when using GPU.
# This speeds up training and reduces VRAM usage.
# It automatically uses float16 on compatible GPUs.
# ------------------------------------------------------------
if cfg["use_gpu"]:
    mixed_precision.set_global_policy("mixed_float16")

# --- THE SECRET SAUCE: OneCycle Scheduler ---
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        pct = min(self.step / self.total_steps, 1.0)
        
        if pct < 0.3:
            lr = self.max_lr * (pct / 0.3)
        else:
            progress = (pct - 0.3) / 0.7
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.max_lr * cosine_decay
            
        lr = max(lr, 1e-6)
        self.model.optimizer.learning_rate = float(lr)
        self.step += 1

# --- MLOps: EXPERIMENT TRACKER ---
class ExperimentTracker(keras.callbacks.Callback):
    def __init__(self, cfg, log_filename="experiment_logs.csv"):
        super().__init__()
        self.cfg = cfg
        # Route it to the logs directory!
        self.log_file = os.path.join(self.cfg['logs_dir'], log_filename)
        self.best_val_acc = 0.0
        self.best_metrics = {}

    def on_train_begin(self, logs=None):
        self.start_time = datetime.datetime.now()
        self.epochs_run = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_run += 1
        # Track the best epoch based on validation accuracy
        current_val_acc = logs.get('val_accuracy', 0)
        if current_val_acc >= self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_metrics = logs.copy()

    def on_train_end(self, logs=None):
        self.end_time = datetime.datetime.now()
        duration = self.end_time - self.start_time

        # Format time and duration
        date_str = self.start_time.strftime("%d/%m/%Y")
        time_start_str = self.start_time.strftime("%H:%M")
        time_end_str = self.end_time.strftime("%H:%M")
        
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours}h {minutes}m"

        # Dynamically fetch dropout if it moved into the kwargs
        kwargs_key = f"{self.cfg['model_type']}_kwargs"
        model_kwargs = self.cfg.get(kwargs_key, {})
        actual_dropout = model_kwargs.get('dropout_rate', self.cfg.get('dropout_rate', 0.0))

        # Assemble the data row
        row = {
            "Profile": self.cfg.get('profile_name', 'unknown').upper(),
            "Date": date_str,
            "Start_Time": time_start_str,
            "End_Time": time_end_str,
            "Duration": duration_str,
            "Model": self.cfg['model_type'].upper(),
            "Pretrained": self.cfg.get('pretrained', False),
            "Epochs_Run": self.epochs_run,
            "Train_Loss": round(self.best_metrics.get('loss', 0), 4),
            "Train_Accuracy": round(self.best_metrics.get('accuracy', 0), 4),
            "Val_Loss": round(self.best_metrics.get('val_loss', 0), 4),
            "Val_Accuracy": round(self.best_metrics.get('val_accuracy', 0), 4),
            "Dropout_Rate": actual_dropout,
            "Learning_Rate": self.cfg.get('learning_rate', 0.001),
            "Batch_Size": self.cfg['batch_size'],
            "Weight_Decay": self.cfg.get('weight_decay', 1e-4),
            "Label_Smoothing": self.cfg.get('label_smoothing', 0.1),
            "History_File": self.cfg.get("history_file", "N/A")
        }

        # FIX: Explicitly enforce the fieldnames list so it never scrambles order
        fieldnames = list(row.keys())
        file_exists = os.path.isfile(self.log_file)
        
        with open(self.log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() 
            writer.writerow(row)
            
        print(f"\n📊 MLOps: Experiment data successfully saved to logs/experiment_logs.csv")

# --- ADVANCED DATA AUGMENTATION ---
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),  
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomTranslation(0.1, 0.1)
], name="data_augmentation_pipeline")

# ------------------------------------------------------------
# Normalize images to [0, 1] range.
# Most neural networks train much better with normalized input.
# ------------------------------------------------------------
def normalize_images(images, labels):
    """
    Convert image pixels from uint8 [0,255] to float32 [0,1].
    """
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

# ------------------------------------------------------------
# MixUp augmentation.
# Combines two random images and their labels.
# This improves generalization significantly for classification.
# ------------------------------------------------------------
def mixup(images, labels, alpha=0.2):
    """
    Apply MixUp augmentation to a batch of images.

    Parameters
    ----------
    images : tensor
        Batch of images.
    labels : tensor
        Corresponding labels.
    alpha : float
        MixUp interpolation strength.

    Returns
    -------
    Mixed images and labels.
    """
    batch_size = tf.shape(images)[0]

    # Sample lambda from Beta distribution
    lam = tf.random.uniform([], 0, 1)

    # Shuffle batch
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels

def get_datasets(cfg):
    """Creates tf.data.Dataset objects with optimized memory and augmentation."""
    print("🚀 Loading Datasets...")
    
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
    
    # 1️⃣ Normalize images first
    train_ds = train_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(normalize_images, num_parallel_calls=AUTOTUNE)

    # 2️⃣ Cache datasets in memory (speeds up repeated epochs)
    if cfg.get("cache_dataset", False):
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # 3️⃣ Apply data augmentation only to training data
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    # 4️⃣ Apply MixUp augmentation (training only)
    train_ds = train_ds.map(mixup, num_parallel_calls=AUTOTUNE)

    # 5️⃣ Prefetch batches to keep GPU busy
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

# --- MLOps: DYNAMIC HISTORY LOGGER (Callback 5) ---
class DynamicHistoryLogger(keras.callbacks.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        # Collect data at the end of every epoch
        logs = logs or {}
        row = {'epoch': epoch + 1} 
        
        # We save the FULL decimal value inside the CSV file!
        for key, value in logs.items():
            row[key] = float(value) 
            
        self.epoch_data.append(row)

    def on_train_end(self, logs=None):
        # Only save if we actually ran at least one epoch
        if not self.epoch_data:
            return
        
        # Find the best validation accuracy achieved during the entire training run
        best_val_acc = max([row.get('val_accuracy', 0.0) for row in self.epoch_data])
        
        # Create the custom filename
        date_str = datetime.datetime.now().strftime("%d-%m-%Y")
        model_name = self.cfg['model_type'].lower()
        
        # Using :.4f rounds the float to 4 decimal places ONLY for the file name string
        filename = f"{model_name}_history_{date_str}_{best_val_acc:.4f}.csv"
        # Route it to the new logs directory!
        filepath = os.path.join(self.cfg['logs_dir'], filename)

        # Save filename in config so ExperimentTracker can access it
        self.cfg["history_file"] = filename
        
        # FIX: Explicitly define columns so it never scrambles
        keys = list(self.epoch_data[0].keys())
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.epoch_data:
                writer.writerow(row)

        print(f"📈 MLOps: Epoch history successfully saved to logs/{filename}")

        # ------------------------------------------------------------
        # Automatically generate training plots
        # ------------------------------------------------------------

        generate_training_plot(
            history_path=filepath,
            model_name=model_name,
            logs_dir=self.cfg["logs_dir"]
        )

# ==========================================
# 🔥 MAIN TRAINING LOOP
# ==========================================
def train():
    # Pass cfg to get_datasets
    train_ds, val_ds = get_datasets(cfg)

    # 1. Construct the exact dictionary key to look for in the YAML
    kwargs_key = f"{cfg['model_type']}_kwargs"
    
    # 2. Extract that specific dictionary from the config
    model_kwargs = cfg.get(kwargs_key, {})

    print(f"🧠 Building Model: {cfg['model_type'].upper()} (Pretrained: {cfg.get('pretrained', False)})...")
    print(f"📐 Architecture Blueprint: {model_kwargs}") 
    
    # 3. The Magic Bridge: Unpack the dictionary directly into the factory using **
    model = get_model(
        model_name=cfg['model_type'],
        input_shape=(cfg['img_size'], cfg['img_size'], 3), 
        num_classes=cfg['num_classes'],
        pretrained=cfg.get('pretrained', False),
        **model_kwargs  
    )

    print("⚙️ Compiling with AdamW and Label Smoothing...")
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=cfg['learning_rate'],
            weight_decay=float(cfg['weight_decay']),
            clipnorm=1.0  # Prevent exploding gradients
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(cfg['label_smoothing'])
        ),
        metrics=['accuracy']
    )

    # Calculate steps
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    total_steps = steps_per_epoch * cfg['epochs']
    # steps_per_epoch = 100000 // cfg['batch_size']
    # total_steps = steps_per_epoch * cfg['epochs']

    save_name = f"{cfg['model_type']}_best.keras"
    checkpoint_path = os.path.join(cfg['models_dir'], save_name)
    
    # ==========================================
    # 🏆 MLOps: GLOBAL HIGH SCORE TRACKER
    # ==========================================
    log_file = os.path.join(cfg['logs_dir'], "experiment_logs.csv")
    historical_best = 0.0
    
    # Read the CSV database to find the highest Val_Accuracy ever recorded for this specific model
    if os.path.isfile(log_file):
        with open(log_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Model', '').lower() == cfg['model_type'].lower():
                    try:
                        acc = float(row.get('Val_Accuracy', 0.0))
                        if acc > historical_best:
                            historical_best = acc
                    except ValueError:
                        pass
                        
    print(f"\n🏆 All-Time High Score for {cfg['model_type'].upper()}: {historical_best:.4f}")
    print("🛡️ New weights will ONLY be saved if they beat this score!\n")

    # We subclass the standard Keras Checkpoint to inject our historical high score and custom UI messages
    class GlobalCheckpoint(keras.callbacks.ModelCheckpoint):
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            self.best = historical_best # Force Keras to remember the all-time high!

        def on_epoch_end(self, epoch, logs=None):
            # Fetch the current epoch's validation accuracy
            current = logs.get(self.monitor)
            if current is not None:
                if current <= self.best:
                    # Custom message when the model fails to beat the record
                    print(f"\n🛑 Not saving: previous model is better (Current: {current:.4f} vs All-Time Best: {self.best:.4f})")
                else:
                    # Custom message when we hit a new high score
                    print(f"\n🌟 NEW HIGH SCORE! ({current:.4f} beats {self.best:.4f}) -> Overwriting model on disk!")
            
            # Now let Keras actually do the background saving logic
            super().on_epoch_end(epoch, logs)

    checkpoint_cb = GlobalCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        monitor='val_accuracy',
        mode='max',
        verbose=0 # <--- Turned off default Keras prints so our custom messages shine
    )
    
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=cfg['patience'], 
        restore_best_weights=True, 
        verbose=1
    )
    
    # Callback 3: Our custom OneCycle scheduler
    one_cycle_cb = OneCycleScheduler(max_lr=cfg['max_lr'], total_steps=total_steps)

    # Callback 4: MLOps Tracker
    tracker_cb = ExperimentTracker(cfg=cfg)

    # Callback 5: Dynamic History Logger
    history_cb = DynamicHistoryLogger(cfg=cfg)

    # ------------------------------------------------------------
    # Callback 6: TensorBoard visualization
    # Allows monitoring training curves, weights, and histograms.
    # ------------------------------------------------------------
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=os.path.join(
            cfg["logs_dir"],
            "tensorboard",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        histogram_freq=1,
    )

    print(f"🔥 Starting Training for up to {cfg['epochs']} Epoch(s)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg['epochs'],
        callbacks=[checkpoint_cb, 
                   early_stop_cb,
                   one_cycle_cb,
                   tracker_cb,
                   history_cb,
                   tensorboard_cb]
    )

    # ------------------------------------------------------------
    # POST-TRAINING EVALUATION PIPELINE
    # ------------------------------------------------------------

    print("\n🧪 Running post-training evaluation...")

    evaluate_model(
        model=model,
        dataset=val_ds,
        class_names=train_ds.class_names,
        model_name=cfg["model_type"],
        logs_dir=cfg["logs_dir"],
        image_paths=val_ds.file_paths
    )


if __name__ == "__main__":
    train()






















'''
import sys
import os
import math
import datetime
import csv

# 1. Take the "blinders" off: Add the project root to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Load Config FIRST so we can set hardware profiles BEFORE TensorFlow boots up
from src.utils import load_config
# Change this to "desktop" when you get to the RTX 3060
cfg = load_config(profile="laptop") 

if not cfg['use_gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("💻 [CONFIG] LAPTOP MODE ACTIVE")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🚀 [CONFIG] DESKTOP MODE ACTIVE")

# 3. Import everything else NOW
import tensorflow as tf
import keras
from src.models import get_model

# --- THE SECRET SAUCE: OneCycle Scheduler ---
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, max_lr, total_steps):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        pct = self.step / self.total_steps
        
        if pct < 0.3:
            lr = self.max_lr * (pct / 0.3)
        else:
            progress = (pct - 0.3) / 0.7
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.max_lr * cosine_decay
            
        lr = max(lr, 1e-6)
        self.model.optimizer.learning_rate = float(lr)
        self.step += 1

# --- MLOps: EXPERIMENT TRACKER ---
class ExperimentTracker(keras.callbacks.Callback):
    def __init__(self, cfg, log_filename="experiment_logs.csv"):
        super().__init__()
        self.cfg = cfg
        # Route it to the logs directory!
        self.log_file = os.path.join(self.cfg['logs_dir'], log_filename)
        self.best_val_acc = 0.0
        self.best_metrics = {}

    def on_train_begin(self, logs=None):
        self.start_time = datetime.datetime.now()
        self.epochs_run = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_run += 1
        # Track the best epoch based on validation accuracy
        current_val_acc = logs.get('val_accuracy', 0)
        if current_val_acc >= self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_metrics = logs.copy()

    def on_train_end(self, logs=None):
        self.end_time = datetime.datetime.now()
        duration = self.end_time - self.start_time

        # Format time and duration
        date_str = self.start_time.strftime("%d/%m/%Y")
        time_start_str = self.start_time.strftime("%H:%M")
        time_end_str = self.end_time.strftime("%H:%M")
        
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours}h {minutes}m"

        # Assemble the data row
        row = {
            "Profile": self.cfg.get('profile_name', 'unknown').upper(),
            "Date": date_str,
            "Start_Time": time_start_str,
            "End_Time": time_end_str,
            "Duration": duration_str,
            "Model": self.cfg['model_type'].upper(),
            "Pretrained": self.cfg.get('pretrained', False),
            "Epochs_Run": self.epochs_run,
            "Train_Loss": round(self.best_metrics.get('loss', 0), 4),
            "Train_Accuracy": round(self.best_metrics.get('accuracy', 0), 4),
            "Val_Loss": round(self.best_metrics.get('val_loss', 0), 4),
            "Val_Accuracy": round(self.best_metrics.get('val_accuracy', 0), 4),
            "Dropout_Rate": self.cfg.get('dropout_rate', 0.3),
            "Learning_Rate": self.cfg.get('learning_rate', 0.001),
            "Batch_Size": self.cfg['batch_size'],
            "Weight_Decay": self.cfg.get('weight_decay', 1e-4),
            "Label_Smoothing": self.cfg.get('label_smoothing', 0.1)
        }

        # FIX: Explicitly enforce the fieldnames list so it never scrambles order
        fieldnames = list(row.keys())
        file_exists = os.path.isfile(self.log_file)
        
        with open(self.log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() 
            writer.writerow(row)
            
        print(f"\n📊 MLOps: Experiment data successfully saved to logs/experiment_logs.csv")

# --- ADVANCED DATA AUGMENTATION ---
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),  
    keras.layers.RandomZoom(0.1),      
], name="data_augmentation_pipeline")

def get_datasets(cfg):
    """Creates tf.data.Dataset objects with optimized memory and augmentation."""
    print("🚀 Loading Datasets...")
    
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

    # 1. Construct the exact dictionary key to look for in the YAML (e.g., "vit" + "_kwargs" = "vit_kwargs")
    kwargs_key = f"{cfg['model_type']}_kwargs"
    
    # 2. Extract that specific dictionary from the config (default to empty {} if it somehow doesn't exist)
    model_kwargs = cfg.get(kwargs_key, {})

    print(f"🧠 Building Model: {cfg['model_type'].upper()} (Pretrained: {cfg.get('pretrained', False)})...")
    print(f"📐 Architecture Blueprint: {model_kwargs}") # Let's print it so we can see the magic happen!
    
    # 3. The Magic Bridge: Unpack the dictionary directly into the factory using **
    model = get_model(
        model_name=cfg['model_type'],
        input_shape=(cfg['img_size'], cfg['img_size'], 3), 
        num_classes=cfg['num_classes'],
        pretrained=cfg.get('pretrained', False),
        **model_kwargs  # <--- THIS IS THE BRIDGE!
    )

    AUTOTUNE = tf.data.AUTOTUNE
    
    # 1. CACHE FIRST
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    # 2. AUGMENT SECOND
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # 3. PREFETCH LAST
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

# --- MLOps: DYNAMIC HISTORY LOGGER (Callback 5) ---
class DynamicHistoryLogger(keras.callbacks.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        # Collect data at the end of every epoch
        logs = logs or {}
        row = {'epoch': epoch + 1} 
        
        # We save the FULL decimal value inside the CSV file!
        for key, value in logs.items():
            row[key] = float(value) 
            
        self.epoch_data.append(row)

    def on_train_end(self, logs=None):
        # Only save if we actually ran at least one epoch
        if not self.epoch_data:
            return
        
        # Find the best validation accuracy achieved during the entire training run
        best_val_acc = max([row.get('val_accuracy', 0.0) for row in self.epoch_data])
        
        # Create the custom filename
        date_str = datetime.datetime.now().strftime("%d-%m-%Y")
        model_name = self.cfg['model_type'].lower()
        
        # Using :.4f rounds the float to 4 decimal places ONLY for the file name string
        filename = f"{model_name}_history_{date_str}_{best_val_acc:.4f}.csv"
        # Route it to the new logs directory!
        filepath = os.path.join(self.cfg['logs_dir'], filename)
        
        # FIX: Explicitly define columns so it never scrambles
        keys = list(self.epoch_data[0].keys())
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.epoch_data:
                writer.writerow(row)
            
        print(f"📈 MLOps: Epoch history successfully saved to logs/{filename}")

def train():
    # Pass cfg to get_datasets
    train_ds, val_ds = get_datasets(cfg)

    print(f"🧠 Building Model: {cfg['model_type'].upper()} (Pretrained: {cfg.get('pretrained', False)})...")
    
    model = get_model(
        model_name=cfg['model_type'],
        input_shape=(cfg['img_size'], cfg['img_size'], 3), 
        num_classes=cfg['num_classes'],
        pretrained=cfg.get('pretrained', False),
        dropout_rate=float(cfg.get('dropout_rate', 0.3)) 
    )

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

    # Calculate steps
    steps_per_epoch = 100000 // cfg['batch_size']
    total_steps = steps_per_epoch * cfg['epochs']

    save_name = f"{cfg['model_type']}_best.keras"
    checkpoint_path = os.path.join(cfg['models_dir'], save_name)
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        monitor='val_accuracy',
        verbose=1
    )
    
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=cfg['patience'], 
        restore_best_weights=True, 
        verbose=1
    )
    
    # Callback 3: Our custom OneCycle scheduler
    one_cycle_cb = OneCycleScheduler(max_lr=cfg['max_lr'], total_steps=total_steps)

    # Callback 4: MLOps Tracker
    tracker_cb = ExperimentTracker(cfg=cfg)

    # Callback 5: Dynamic History Logger
    history_cb = DynamicHistoryLogger(cfg=cfg)

    print(f"🔥 Starting Training for up to {cfg['epochs']} Epoch(s)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg['epochs'],
        callbacks=[checkpoint_cb, early_stop_cb, one_cycle_cb, tracker_cb, history_cb]
    )
    
if __name__ == "__main__":
    train()


'''