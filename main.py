"""
main.py
--------
Entry point for the Handwritten Character Recognition project.

Running this script will:
    1. Load and preprocess the MNIST dataset.
    2. Build the CNN model.
    3. Train the model for up to 15 epochs with early stopping.
    4. Save the trained model to models/handwritten_model.keras
    5. Generate and save training accuracy/loss plots.
    6. Evaluate the model on the test set (accuracy, classification
       report, confusion matrix).

Usage
-----
    python main.py
"""

from src.train import train_model
from src.evaluate import evaluate_model


def main():
    """Run the full training + evaluation pipeline end to end."""
    print("\n" + "#" * 60)
    print("# Handwritten Digit Recognition")
    print("#" * 60 + "\n")

    # --- Train the model (also saves the model and training plots) ---
    model, history, (x_test, y_test) = train_model()

    # --- Evaluate the trained model on the test set ---
    print("\n" + "#" * 60)
    print("# Model Evaluation")
    print("#" * 60 + "\n")
    results = evaluate_model(model, x_test, y_test)

    print("\n" + "#" * 60)
    print("# Pipeline complete!")
    print("#" * 60)
    print(f"Final Test Accuracy: {results['test_accuracy'] * 100:.2f}%")
    print("Model saved in:   models/handwritten_model.keras")
    print("Plots saved in:   plots/")


if __name__ == "__main__":
    main()