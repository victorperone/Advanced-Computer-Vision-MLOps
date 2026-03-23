"""
Optuna Hyperparameter Search for TinyImageNet Project using Optuna
-----------------------------------------------------

This script automatically searches for the best hyperparameters
for the models defined in the project.

Industry-style features:
- automatic experiment tracking
- pruning of bad trials
- reproducible experiments

Key Features
------------
- model-agnostic search
- dynamic architecture search
- separate studies per model
- integration with project config system


To visualize the results open optuna-dashboard:

S optuna-dashboard sqlite:///optuna.db

"""

import sys
import os

import optuna
import tensorflow as tf
import keras

from src.utils import load_config
from src.models import get_model
from src.train import get_datasets


def build_search_space(trial, model_type):
    """
    Define the hyperparameter search space dynamically
    depending on the model architecture.
    """

    # -----------------------------------------
    # Shared parameters (all models)
    # -----------------------------------------

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        5e-3,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )

    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.1,
        0.5,
    )

    params = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
    }

    # -----------------------------------------
    # Vision Transformer search space
    # -----------------------------------------

    if model_type == "vit":

        params.update({
            "patch_size": trial.suggest_categorical(
                "patch_size", [4, 8]
            ),
            "projection_dim": trial.suggest_categorical(
                "projection_dim", [64, 128, 192]
            ),
            "num_heads": trial.suggest_categorical(
                "num_heads", [4, 8, 12]
            ),
            "transformer_layers": trial.suggest_int(
                "transformer_layers",
                4,
                10,
            ),
        })

    # -----------------------------------------
    # ResNet search space
    # -----------------------------------------

    elif model_type == "resnet":

        params.update({
            "base_filters": trial.suggest_categorical(
                "base_filters", [32, 64, 96]
            )
        })

    # -----------------------------------------
    # EfficientNet search space
    # -----------------------------------------

    elif model_type == "efficientnet":

        # EfficientNet mainly benefits from dropout tuning
        pass

    # -----------------------------------------
    # ConvNeXt search space
    # -----------------------------------------

    elif model_type == "convnext":

        pass

    return params


def objective(trial, model_type):
    """
    Optuna objective function.

    Each trial trains a model using different
    hyperparameters and returns validation accuracy.
    """

    cfg = load_config(profile="desktop")

    # -----------------------------------------
    # Build search space
    # -----------------------------------------

    params = build_search_space(trial, model_type)

    learning_rate = params.pop("learning_rate")
    weight_decay = params.pop("weight_decay")

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------

    train_ds, val_ds = get_datasets(cfg)

    # -----------------------------------------
    # Build model
    # -----------------------------------------

    model = get_model(
        model_name=model_type,
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", False),
        **params
    )

    # -----------------------------------------
    # Compile model
    # -----------------------------------------

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg["label_smoothing"]
        ),
        metrics=["accuracy"],
    )

    # -----------------------------------------
    # Short training for search
    # -----------------------------------------

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=0,
    )

    val_acc = max(history.history["val_accuracy"])

    return val_acc


def run_search(model_type="vit", trials=30):
    """
    Run hyperparameter optimization for a specific model.
    """

    study_name = f"{model_type}_tinyimagenet_search"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, model_type),
        n_trials=trials,
    )

    print("\nBest Trial")
    print("Accuracy:", study.best_value)
    print("Params:", study.best_params)


if __name__ == "__main__":

    # Example search
    run_search(model_type="vit", trials=30)

"""
Edit bottom of script:
run_search(model_type="efficientnet", trials=30)
run_search(model_type="convnext", trials=30)
run_search(model_type="resnet", trials=30)
"""