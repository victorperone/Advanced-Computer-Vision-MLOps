"""
utils.py
========

Shared utility functions used across train.py, fine_tune.py, and ensemble.py.

Functions
---------
load_config     -- Load and merge config.yaml with a hardware profile.
set_global_seed -- Pin all RNG sources for reproducible experiments.

PEP 8 notes
-----------
* Max line length 79 characters.
* Two blank lines between top-level definitions.
* Docstrings follow NumPy style.
"""

import os
import random

import numpy as np
import yaml


def load_config(profile="laptop"):
    """
    Load ``config.yaml`` and merge the dataset section with a hardware profile.

    The YAML file has three top-level keys:
        dataset  — shared paths and image dimensions (used by all profiles).
        laptop   — CPU-optimised hyperparameters and tiny model configs.
        desktop  — GPU-optimised hyperparameters and full model configs.

    This function merges ``dataset`` into the chosen profile so callers
    get a single flat dict with all keys available.

    Parameters
    ----------
    profile : str
        Either ``"laptop"`` or ``"desktop"``.  Must match a top-level key
        in config.yaml.

    Returns
    -------
    dict
        Merged configuration dictionary.  Includes the ``profile_name`` key
        so callbacks can log which profile was active.

    Raises
    ------
    FileNotFoundError
        If config.yaml is not found relative to the project root.
    KeyError
        If ``profile`` does not exist as a top-level key in config.yaml.
    """
    # Locate config.yaml relative to THIS file (src/utils.py → project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if profile not in config:
        raise KeyError(
            f"Profile '{profile}' not found in config.yaml. "
            f"Available profiles: {[k for k in config if k != 'dataset']}"
        )

    # Start with shared dataset settings, then overlay the profile values.
    # Profile keys win on collision (e.g. if both define something unexpected).
    active_config = dict(config["dataset"])
    active_config.update(config[profile])

    # Record which profile is active — used by ExperimentTracker for logging
    active_config["profile_name"] = profile

    # Resolve absolute paths for model and log directories so scripts work
    # regardless of which directory they are invoked from.
    active_config["models_dir"] = os.path.join(base_dir, "models")
    active_config["logs_dir"] = os.path.join(base_dir, "logs")

    os.makedirs(active_config["models_dir"], exist_ok=True)
    os.makedirs(active_config["logs_dir"], exist_ok=True)

    return active_config


def set_global_seed(seed=42):
    """
    Pin every random number generator used by the training stack.

    Why is this needed?
        A deep learning training run involves at least four independent RNG
        sources that are all seeded from the OS clock by default:

        1. Python ``random`` module — used by some data-loading helpers.
        2. NumPy — used by sklearn metrics and some augmentation libs.
        3. TensorFlow / Keras — op-level ops (weight init, dropout masks,
           etc.).  Keras wraps this via ``keras.utils.set_random_seed``.
        4. The ``tf.data`` shuffle buffer — controlled separately via the
           ``seed`` argument to ``image_dataset_from_directory`` (already
           set to 42 in get_datasets), but the global TF seed is a
           belt-and-braces safety net.

        Without pinning all four, two runs on the same machine with the
        same config will produce different weight initialisations, different
        shuffle orders (beyond the first epoch), and different dropout masks
        — all of which accumulate to materially different final accuracies.

    Reproducibility caveats
        * CPU training: fully deterministic after this call.
        * GPU training: cuDNN uses non-deterministic parallel reductions
          for some ops (e.g. atomic adds in conv backward pass).  You can
          force determinism with ``tf.config.experimental.enable_op_determinism()``
          but this can slow GPU training by 10–30%.  For research you
          typically accept a small GPU variance and report mean ± std over
          3 runs instead.

    Parameters
    ----------
    seed : int
        The seed value.  42 is the conventional default; any integer works.
        Store this in config.yaml under the ``seed`` key so it is logged
        with every experiment.

    Returns
    -------
    None

    Examples
    --------
    >>> from src.utils import set_global_seed
    >>> set_global_seed(42)
    🌱 Global seed set to 42 (Python, NumPy, TensorFlow/Keras)
    """
    # 1. Python built-in random module
    random.seed(seed)

    # 2. NumPy — used by sklearn metrics in evaluation.py
    np.random.seed(seed)

    # 3. TensorFlow + Keras (single call pins both)
    # keras.utils.set_random_seed sets:
    #   - Python random seed
    #   - NumPy seed
    #   - TF global seed (tf.random.set_seed)
    # We call it explicitly for belt-and-braces even though we already set
    # Python and NumPy above — it ensures the TF graph-level seed is also set.
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        # Fallback if running without Keras (e.g. standalone utils import)
        import tensorflow as tf
        tf.random.set_seed(seed)

    # Set the environment variable that some TF internal ops read
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"🌱 Global seed set to {seed} (Python, NumPy, TensorFlow/Keras)")
