"""
train.py
---------
Trains the CNN model on the MNIST dataset, saves the trained model to
disk, and generates/saves plots of training & validation accuracy/loss.

Can be run standalone:
    python src/train.py
"""
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

import os
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt

# Allow running this file both as part of the `src` package (via main.py)
# and directly as a standalone script (`python src/train.py`).
try:
    from src.data_loader import get_mnist_dataset
    from src.model import build_cnn_model
except ImportError:
    from data_loader import get_mnist_dataset
    from model import build_cnn_model



# --- Project paths ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")
MODEL_PATH = os.path.join(MODELS_DIR, "handwritten_model.keras")

# --- Training hyperparameters ---
EPOCHS = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1  # 10% of training data held out for validation


def plot_training_history(history, plots_dir=PLOTS_DIR):
    """
    Plot and save training/validation accuracy and loss curves.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned by model.fit().
    plots_dir : str
        Directory in which to save the generated plot images.
    """
    os.makedirs(plots_dir, exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    # --- Accuracy plot (training vs validation) ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, "b-o", label="Training Accuracy")
    plt.plot(epochs_range, val_acc, "r-o", label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    accuracy_path = os.path.join(plots_dir, "accuracy_plot.png")
    plt.savefig(accuracy_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy plot to: {accuracy_path}")

    # --- Loss plot (training vs validation) ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, "b-o", label="Training Loss")
    plt.plot(epochs_range, val_loss, "r-o", label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(plots_dir, "loss_plot.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to: {loss_path}")


def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, save_model=True):
    """
    Full training pipeline: load data, build model, train, save model
    and plots.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size used during training.
    save_model : bool
        Whether to save the trained model to MODEL_PATH.

    Returns
    -------
    model : tf.keras.Model
        The trained Keras model.
    history : tf.keras.callbacks.History
        Training history (accuracy/loss per epoch).
    (x_test, y_test) : tuple of numpy arrays
        Preprocessed test set, returned for convenience so callers (e.g.
        main.py) can immediately run evaluation without reloading data.
    """
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
    

    from tensorflow.keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )

    callbacks = [
        early_stop,
        checkpoint,
        reduce_lr
    ]
    print("=" * 60)
    print("STEP 1/4: Loading and preprocessing MNIST dataset...")
    print("=" * 60)
    (x_train, y_train), (x_test, y_test) = get_mnist_dataset()
    print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    print("\n" + "=" * 60)
    print("STEP 2/4: Building the CNN model...")
    print("=" * 60)
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    model.summary()

    print("\n" + "=" * 60)
    print(f"STEP 3/4: Training the model for {epochs} epochs...")
    print("=" * 60)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=2,
    )

    print("\n" + "=" * 60)
    print("STEP 4/4: Saving model and training plots...")
    print("=" * 60)
    if save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"Model saved to: {MODEL_PATH}")

    plot_training_history(history)

    return model, history, (x_test, y_test)


if __name__ == "__main__":
    train_model()