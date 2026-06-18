"""
data_loader.py
---------------
Handles loading, preprocessing (normalization + reshaping), and basic
inspection of the MNIST handwritten digits dataset.

The MNIST dataset consists of 70,000 grayscale images of handwritten
digits (0-9), each of size 28x28 pixels:
    - 60,000 training images
    - 10,000 test images

Functions
---------
load_mnist_data()
    Loads the raw MNIST dataset (train/test split) as numpy arrays.
preprocess_data(x_train, y_train, x_test, y_test)
    Normalizes pixel values to [0, 1] and reshapes images to (28, 28, 1)
    for use with a Conv2D-based CNN. Labels are kept as integer class
    indices (suitable for `sparse_categorical_crossentropy`).
get_mnist_dataset()
    Convenience wrapper that loads and fully preprocesses the dataset
    in a single call.
"""

import os
import numpy as np
import tensorflow as tf

# Path to a local fallback copy of MNIST (used if the online download
# via tf.keras.datasets.mnist fails, e.g. due to no internet access).
# This file is stored in the exact format produced by
# `tf.keras.datasets.mnist.load_data()` internally (an .npz archive with
# keys: x_train, y_train, x_test, y_test).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MNIST_PATH = os.path.join(_PROJECT_ROOT, "data", "mnist.npz")


def load_mnist_data():
    """
    Load the raw MNIST dataset as numpy arrays.

    Attempts to download the dataset via TensorFlow/Keras first. If that
    fails (e.g. no internet connection), falls back to a local copy
    bundled in the `data/` directory of this project.

    Returns
    -------
    (x_train, y_train), (x_test, y_test) : tuple of numpy arrays
        x_train : uint8 array of shape (60000, 28, 28)
        y_train : uint8 array of shape (60000,)
        x_test  : uint8 array of shape (10000, 28, 28)
        y_test  : uint8 array of shape (10000,)
    """
    try:
        # Standard approach: Keras downloads/caches MNIST automatically.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print("MNIST dataset loaded successfully via tf.keras.datasets.")
    except Exception as exc:  # noqa: BLE001 - we want a broad fallback here
        print(f"Could not download MNIST via tf.keras.datasets ({exc}).")
        if os.path.exists(_LOCAL_MNIST_PATH):
            print(f"Falling back to local copy at: {_LOCAL_MNIST_PATH}")
            with np.load(_LOCAL_MNIST_PATH) as data:
                x_train, y_train = data["x_train"], data["y_train"]
                x_test, y_test = data["x_test"], data["y_test"]
        else:
            raise RuntimeError(
                "MNIST could not be downloaded and no local fallback was "
                f"found at {_LOCAL_MNIST_PATH}. Please connect to the "
                "internet or place a valid mnist.npz file in the data/ "
                "directory."
            ) from exc

    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocess MNIST images and labels for CNN training.

    Steps performed:
        1. Cast pixel values to float32 and normalize to the [0, 1] range
           (raw pixel values are in [0, 255]).
        2. Reshape images from (N, 28, 28) to (N, 28, 28, 1), adding an
           explicit single-channel (grayscale) dimension required by
           Conv2D layers.

    Labels are returned unchanged as integer class indices (0-9), which
    is the format expected when training with
    `sparse_categorical_crossentropy` loss.

    Parameters
    ----------
    x_train, x_test : numpy arrays of shape (N, 28, 28)
    y_train, y_test : numpy arrays of shape (N,)

    Returns
    -------
    (x_train, y_train), (x_test, y_test) : preprocessed numpy arrays
        x_train : float32 array of shape (60000, 28, 28, 1), values in [0, 1]
        x_test  : float32 array of shape (10000, 28, 28, 1), values in [0, 1]
        y_train : uint8 array of shape (60000,)
        y_test  : uint8 array of shape (10000,)
    """
    # Normalize pixel values from [0, 255] to [0, 1].
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to add the channel dimension expected by Conv2D: (28, 28, 1).
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)


def get_mnist_dataset():
    """
    Load and fully preprocess the MNIST dataset in one call.

    Returns
    -------
    (x_train, y_train), (x_test, y_test) : preprocessed numpy arrays
        Ready to be fed directly into a Keras CNN model.
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    return preprocess_data(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    # Quick sanity check when running this module directly.
    (x_train, y_train), (x_test, y_test) = get_mnist_dataset()
    print(f"x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"x_test shape:  {x_test.shape}, dtype: {x_test.dtype}")
    print(f"y_test shape:  {y_test.shape}, dtype: {y_test.dtype}")
    print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
    print(f"Unique labels: {sorted(set(y_train.tolist()))}")