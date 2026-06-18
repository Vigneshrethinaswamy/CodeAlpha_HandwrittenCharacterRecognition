"""
predict.py
-----------
Utility for predicting the digit contained in a single handwritten
character image using the trained CNN model.

Supports:
    - Image files (.png, .jpg, .jpeg, .bmp, etc.) loaded via Pillow.
    - Raw numpy arrays (e.g. an MNIST test sample).

Usage (command line)
---------------------
    python src/predict.py path/to/digit_image.png

Usage (as a module)
---------------------
    from src.predict import predict_digit
    label, confidence = predict_digit("path/to/digit_image.png")
"""

import os
import sys
import numpy as np
from PIL import Image
from tensorflow import keras

# --- Project paths ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "handwritten_model.keras")

IMG_SIZE = 28  # MNIST images are 28x28 pixels


def _load_model(model_path=MODEL_PATH):
    """Load the trained Keras model from disk, raising a clear error if missing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Please run `python src/train.py` (or `python main.py`) first "
            "to train and save the model."
        )
    return keras.models.load_model(model_path)


def preprocess_image(image_path):
    """
    Load an arbitrary image file from disk and preprocess it to match
    the format the CNN was trained on: 28x28 grayscale, normalized to
    [0, 1], shaped as (1, 28, 28, 1).

    The function automatically:
        - Converts the image to grayscale ('L' mode).
        - Resizes it to 28x28 pixels.
        - Inverts colors if needed, so the digit is white-on-black like
          MNIST (MNIST digits are bright strokes on a dark background).
          This heuristic check is based on average pixel brightness: if
          the image background looks lighter than the foreground (i.e.
          a typical photo of pen/pencil on white paper), colors are
          inverted to match MNIST's convention.

    Parameters
    ----------
    image_path : str
        Path to the image file to preprocess.

    Returns
    -------
    numpy array of shape (1, 28, 28, 1), dtype float32, values in [0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")

    # Convert to grayscale and resize to MNIST's 28x28 resolution.
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img).astype("float32")

    # Heuristic: MNIST digits are light strokes on a dark background.
    # If the average pixel value is high (i.e. mostly light/white
    # background, as in a typical scanned/photographed digit on paper),
    # invert so it matches the MNIST convention the model was trained on.
    if img_array.mean() > 127:
        img_array = 255.0 - img_array

    # Normalize to [0, 1], matching training-time preprocessing.
    img_array = img_array / 255.0

    # Reshape to (1, 28, 28, 1): batch of one, single channel.
    img_array = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 1))

    return img_array


def predict_digit(image_path, model=None, model_path=MODEL_PATH):
    """
    Predict the digit (0-9) contained in a single image file.

    Parameters
    ----------
    image_path : str
        Path to the image file containing a single handwritten digit.
    model : tf.keras.Model, optional
        A pre-loaded model. If None, the model is loaded from
        `model_path`. Useful to avoid reloading the model repeatedly
        when predicting many images in a loop.
    model_path : str
        Path to the saved .keras model file (only used if `model` is
        None).

    Returns
    -------
    predicted_label : int
        The predicted digit (0-9).
    confidence : float
        The model's confidence (softmax probability) for the predicted
        class, in the range [0, 1].
    """
    if model is None:
        model = _load_model(model_path)

    processed_image = preprocess_image(image_path)
    probabilities = model.predict(processed_image, verbose=0)[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])

    return predicted_label, confidence


def predict_from_array(image_array, model=None, model_path=MODEL_PATH):
    """
    Predict the digit for an in-memory image array (e.g. a sample taken
    directly from the MNIST dataset), rather than a file on disk.

    Parameters
    ----------
    image_array : numpy array
        Image data. Accepted shapes: (28, 28), (28, 28, 1), or already
        batched (1, 28, 28, 1). Values may be in [0, 255] or [0, 1];
        this function normalizes automatically if needed.
    model : tf.keras.Model, optional
        A pre-loaded model. If None, the model is loaded from
        `model_path`.
    model_path : str
        Path to the saved .keras model file (only used if `model` is
        None).

    Returns
    -------
    predicted_label : int
        The predicted digit (0-9).
    confidence : float
        The model's confidence for the predicted class.
    """
    if model is None:
        model = _load_model(model_path)

    arr = np.array(image_array).astype("float32")

    # Normalize to [0, 1] if values appear to be in the [0, 255] range.
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Ensure shape (1, 28, 28, 1).
    if arr.ndim == 2:
        arr = arr.reshape((1, IMG_SIZE, IMG_SIZE, 1))
    elif arr.ndim == 3:
        arr = arr.reshape((1, IMG_SIZE, IMG_SIZE, 1))

    probabilities = model.predict(arr, verbose=0)[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])

    return predicted_label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    label, confidence = predict_digit(image_path)
    print(f"Image: {image_path}")
    print(f"Predicted digit: {label}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")