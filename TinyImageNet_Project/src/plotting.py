"""
Plotting utilities for training history.

Generates accuracy and loss plots automatically
from training history CSV files.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_training_plot(history_path, model_name, logs_dir):
    """
    Generate accuracy and loss plots from a training history CSV.

    Parameters
    ----------
    history_path : str
        Path to the training history CSV file.

    model_name : str
        Name of the model architecture.

    logs_dir : str
        Base logs directory from configuration.
    """

    df = pd.read_csv(history_path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------
    # Accuracy plot
    # ------------------------------------------------------------
    ax[0].plot(df["epoch"], df["accuracy"], label="train")
    ax[0].plot(df["epoch"], df["val_accuracy"], label="validation")

    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # ------------------------------------------------------------
    # Loss plot
    # ------------------------------------------------------------
    ax[1].plot(df["epoch"], df["loss"], label="train")
    ax[1].plot(df["epoch"], df["val_loss"], label="validation")

    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    # ------------------------------------------------------------
    # Create model-specific plot directory
    # logs/plots/<model_name>/
    # ------------------------------------------------------------
    plot_dir = os.path.join(logs_dir, "plots", model_name)
    os.makedirs(plot_dir, exist_ok=True)

    plot_name = os.path.basename(history_path).replace(".csv", ".png")
    output_path = os.path.join(plot_dir, plot_name)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"📈 Training plot saved → {output_path}")