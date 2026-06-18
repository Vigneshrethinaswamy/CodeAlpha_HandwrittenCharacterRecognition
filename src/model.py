"""
model.py
---------
Defines the Convolutional Neural Network (CNN) architecture used for
handwritten digit recognition on the MNIST dataset.

Architecture
------------
Input (28, 28, 1)
  -> Conv2D(32, 3x3) + ReLU
  -> MaxPooling2D(2x2)
  -> Conv2D(64, 3x3) + ReLU
  -> MaxPooling2D(2x2)
  -> Flatten
  -> Dense(128) + ReLU
  -> Dropout(0.5)
  -> Dense(10) + Softmax
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build and compile the CNN model for handwritten digit classification.

    Parameters
    ----------
    input_shape : tuple, default (28, 28, 1)
        Shape of a single input image (height, width, channels).
    num_classes : int, default 10
        Number of output classes (digits 0-9).

    Returns
    -------
    model : tf.keras.Model
        A compiled Keras Sequential model ready for training.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="input_image"),

            # --- First convolutional block ---
            # 32 filters of size 3x3 learn low-level features such as
            # edges and strokes.
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2d_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),

            # --- Second convolutional block ---
            # 64 filters learn higher-level, more abstract features
            # built on top of the first block's outputs.
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),

            # --- Classifier head ---
            # Flatten the 2D feature maps into a 1D feature vector.
            layers.Flatten(name="flatten"),

            # Fully connected layer to combine extracted features.
            layers.Dense(128, activation="relu", name="dense_1"),

            # Dropout regularization to reduce overfitting by randomly
            # disabling 50% of neurons during each training step.
            layers.Dropout(0.5, name="dropout"),

            # Output layer: 10 units (one per digit class), softmax
            # activation converts logits into class probabilities.
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="mnist_cnn",
    )

    # Compile the model:
    #   - Adam optimizer: adaptive learning rate, works well by default.
    #   - sparse_categorical_crossentropy: appropriate since labels are
    #     integer class indices (0-9) rather than one-hot vectors.
    #   - accuracy: primary metric to track during training/evaluation.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print its architecture.
    model = build_cnn_model()
    model.summary()