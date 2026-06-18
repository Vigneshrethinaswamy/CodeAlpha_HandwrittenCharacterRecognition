"""
evaluate.py
------------
Evaluates a trained MNIST CNN model: computes test accuracy/loss,
prints a detailed classification report, and plots/saves a confusion
matrix heatmap.

Can be run standalone (loads the saved model from models/):
    python src/evaluate.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

try:
    from src.data_loader import get_mnist_dataset
except ImportError:
    from data_loader import get_mnist_dataset

# --- Project paths ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")
MODEL_PATH = os.path.join(MODELS_DIR, "handwritten_model.keras")

CLASS_NAMES = [str(i) for i in range(10)]


def evaluate_model(model, x_test, y_test, plots_dir=PLOTS_DIR):
    """
    Evaluate a trained model on the test set: prints accuracy/loss, a
    full classification report, and saves a confusion matrix plot.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    x_test : numpy array of shape (N, 28, 28, 1)
        Preprocessed test images.
    y_test : numpy array of shape (N,)
        True integer class labels (0-9).
    plots_dir : str
        Directory in which to save the confusion matrix plot.

    Returns
    -------
    dict
        Dictionary containing 'test_loss', 'test_accuracy', and the
        raw 'predictions' (predicted class indices) for further use.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # --- Overall test loss & accuracy ---
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # --- Generate predictions ---
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- Classification report (precision, recall, f1-score per class) ---
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Save the classification report to a text file alongside the plots.
    report_path = os.path.join(plots_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("MNIST CNN - Test Set Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n\n")
        f.write(report)
    print(f"Saved classification report to: {report_path}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
    )
    plt.title("Confusion Matrix - MNIST Digit Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "predictions": y_pred,
    }


def load_and_evaluate(model_path=MODEL_PATH):
    """
    Convenience function: load a saved model from disk and evaluate it
    on the MNIST test set. Used when running this script standalone.

    Parameters
    ----------
    model_path : str
        Path to the saved .keras model file.

    Returns
    -------
    dict
        See `evaluate_model` return value.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Please run `python src/train.py` (or `python main.py`) first."
        )

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    print("Loading and preprocessing MNIST test data...")
    (_, _), (x_test, y_test) = get_mnist_dataset()

    return evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    load_and_evaluate()