"""
evaluation.py

Post-training evaluation pipeline.

This module automatically analyzes model predictions and produces
artifacts that can later be explored in notebooks.

Generated artifacts:

logs/evaluation/<model_name>/
    predictions.csv
    misclassified_images.csv
    hardest_classes.csv
    most_confused_pairs.csv
    classification_report.txt
    confusion_matrix.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def evaluate_model(model, dataset, class_names, model_name, logs_dir, image_paths):
    """
    Run evaluation on validation dataset and generate artifacts.

    Parameters
    ----------
    model : keras.Model
        Trained neural network

    dataset : tf.data.Dataset
        Validation dataset

    class_names : list
        Dataset class labels

    model_name : str
        Model architecture name

    logs_dir : str
        Root logs directory

    image_paths : list
        Original image file paths from dataset
    """

    print("\n🔎 Starting evaluation pipeline...")

    # ------------------------------------------------------------
    # Storage containers
    # ------------------------------------------------------------

    y_true = []
    y_pred = []
    confidences = []
    paths = []

    path_index = 0

    # ------------------------------------------------------------
    # Iterate through validation dataset
    # ------------------------------------------------------------

    for images, labels in dataset:

        batch_size = images.shape[0]

        predictions = model.predict(images, verbose=0)

        true_classes = np.argmax(labels.numpy(), axis=1)
        pred_classes = np.argmax(predictions, axis=1)

        conf = np.max(predictions, axis=1)

        y_true.extend(true_classes)
        y_pred.extend(pred_classes)
        confidences.extend(conf)

        # Match images to their original file paths
        batch_paths = image_paths[path_index:path_index + batch_size]
        paths.extend(batch_paths)

        path_index += batch_size

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)

    # ------------------------------------------------------------
    # Create evaluation directory
    # ------------------------------------------------------------

    eval_dir = os.path.join(logs_dir, "evaluation", model_name)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"📂 Saving evaluation artifacts to {eval_dir}")

    # ------------------------------------------------------------
    # 1️⃣ Save predictions.csv
    # ------------------------------------------------------------

    predictions_df = pd.DataFrame({
        "image_path": paths,
        "true_label_index": y_true,
        "predicted_label_index": y_pred,
        "confidence": confidences
    })

    predictions_df["true_label"] = predictions_df["true_label_index"].apply(
        lambda x: class_names[x]
    )

    predictions_df["predicted_label"] = predictions_df["predicted_label_index"].apply(
        lambda x: class_names[x]
    )

    predictions_path = os.path.join(eval_dir, "predictions.csv")

    predictions_df.to_csv(predictions_path, index=False)

    print("📄 predictions.csv saved")

    # ------------------------------------------------------------
    # 2️⃣ Save misclassified_images.csv
    # ------------------------------------------------------------

    errors = predictions_df[
        predictions_df.true_label_index != predictions_df.predicted_label_index
    ].copy()

    # Sort by confidence (most confident wrong predictions first)
    errors = errors.sort_values("confidence", ascending=False)

    misclassified_path = os.path.join(eval_dir, "misclassified_images.csv")

    errors.to_csv(misclassified_path, index=False)

    print("📄 misclassified_images.csv saved")

    # ------------------------------------------------------------
    # 3️⃣ Hardest classes analysis
    # ------------------------------------------------------------

    print("📉 Computing hardest classes...")

    class_accuracy = []

    for i, class_name in enumerate(class_names):

        mask = y_true == i
        total = np.sum(mask)

        if total == 0:
            continue

        correct = np.sum((y_true == i) & (y_pred == i))
        accuracy = correct / total

        class_accuracy.append({
            "class": class_name,
            "accuracy": accuracy,
            "samples": int(total)
        })

    hardest_df = pd.DataFrame(class_accuracy)

    hardest_df = hardest_df.sort_values("accuracy")

    hardest_path = os.path.join(eval_dir, "hardest_classes.csv")

    hardest_df.to_csv(hardest_path, index=False)

    print("📄 hardest_classes.csv saved")

    # ------------------------------------------------------------
    # 4️⃣ Most confused class pairs
    # ------------------------------------------------------------

    print("🔀 Computing confused class pairs...")

    errors_mask = y_true != y_pred

    error_true = y_true[errors_mask]
    error_pred = y_pred[errors_mask]

    confusion_pairs = pd.DataFrame({
        "true_class": [class_names[i] for i in error_true],
        "predicted_class": [class_names[i] for i in error_pred]
    })

    pair_counts = (
        confusion_pairs
        .groupby(["true_class", "predicted_class"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    pairs_path = os.path.join(eval_dir, "most_confused_pairs.csv")

    pair_counts.to_csv(pairs_path, index=False)

    print("📄 most_confused_pairs.csv saved")

    # ------------------------------------------------------------
    # 5️⃣ Classification report
    # ------------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    report_path = os.path.join(eval_dir, "classification_report.txt")

    with open(report_path, "w") as f:
        f.write(report)

    print("📄 classification_report.txt saved")

    # ------------------------------------------------------------
    # 6️⃣ Confusion matrix
    # ------------------------------------------------------------

    print("📊 Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=False,
        yticklabels=False
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_path = os.path.join(eval_dir, "confusion_matrix.png")

    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    print("📄 confusion_matrix.png saved")

    print("\n✅ Evaluation pipeline completed successfully!")
