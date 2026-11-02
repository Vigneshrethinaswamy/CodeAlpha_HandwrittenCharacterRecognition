# Handwritten Digit Recognition

A complete, end-to-end **Handwritten Digit Recognition** system implemented with **TensorFlow/Keras**, trained on the classic **MNIST** dataset. The project covers data loading and preprocessing, CNN model design, training, evaluation, visualization, and single-image inference — all organized into a clean, modular, production-style codebase.

## Overview

This project trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from 28×28 grayscale images. The training script is configured for up to **15 epochs**, with early stopping and checkpointing. The README's recorded test report is shown below; it should be treated as a previously recorded result until reproduced from a clean run of the current code.

| Metric | Value |
|---|---|
| Test Accuracy | **99.30%** |
| Test Loss | 0.0217 |
| Training Epochs | 15 (with EarlyStopping) |
| Total Parameters | 225,034 |

## Project Structure

```
Handwritten-Digit-Recognition/
│
├── data/                      # Local MNIST dataset cache (mnist.npz)
├── models/                    # Saved trained model (handwritten_model.keras)
├── plots/                     # Generated training/evaluation plots and reports
├── notebooks/                 # Jupyter notebook for interactive exploration
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Load, normalize, and reshape MNIST data
│   ├── model.py                # CNN architecture definition
│   ├── train.py                 # Training pipeline + history plots
│   ├── evaluate.py             # Evaluation: accuracy, classification report, confusion matrix
│   └── predict.py               # Single-image digit prediction utility
│
├── requirements.txt
├── README.md
└── main.py                     # End-to-end pipeline entry point
```

## Model Architecture

The CNN is built using `tf.keras.Sequential` with the following layers:

```
Input (28, 28, 1)
  → Conv2D(32, 3x3) + ReLU
  → MaxPooling2D(2x2)
  → Conv2D(64, 3x3) + ReLU
  → MaxPooling2D(2x2)
  → Flatten
  → Dense(128) + ReLU
  → Dropout(0.5)
  → Dense(10) + Softmax
```

**Compilation settings:**
- Optimizer: `Adam`
- Loss: `sparse_categorical_crossentropy` (labels are integer class indices, not one-hot)
- Metric: `accuracy`

## Setup

### 1. Clone / copy the project and navigate into it

```bash
cd Handwritten-Digit-Recognition
```

### 2. Install dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Dataset

The dataset is loaded automatically by `src/data_loader.py` via `tf.keras.datasets.mnist`, which downloads MNIST on first run and caches it locally. **If your environment has no internet access** (e.g. an offline machine or a sandboxed CI runner), the loader automatically falls back to a local copy bundled at `data/mnist.npz` — no extra steps required. This makes the project runnable out-of-the-box in restricted network environments.

## Usage

### Run the full pipeline (train + evaluate)

```bash
python main.py
```

This will:
1. Load and preprocess the MNIST dataset (normalize to `[0, 1]`, reshape to `(28, 28, 1)`).
2. Build the CNN model.
3. Train for up to 15 epochs (with a 10% validation split; early stopping may stop sooner).
4. Save the trained model to `models/handwritten_model.keras`.
5. Generate and save training/validation accuracy and loss plots to `plots/`.
6. Evaluate the model on the test set, printing accuracy, a full classification report, and saving a confusion matrix plot.

### Run individual steps

Each module can also be run standalone:

```bash
# Just inspect the data pipeline
python src/data_loader.py

# Just inspect the model architecture
python src/model.py

# Train only (saves model + accuracy/loss plots)
python src/train.py

# Evaluate a previously saved model (loads models/handwritten_model.keras)
python src/evaluate.py

# Predict a single digit from an image file
python src/predict.py path/to/your_digit_image.png
```

### Predicting a single digit (programmatic usage)

```python
from src.predict import predict_digit

label, confidence = predict_digit("path/to/digit.png")
print(f"Predicted digit: {label} (confidence: {confidence:.2%})")
```

`predict.py` automatically converts the input image to grayscale, resizes it to 28×28, normalizes pixel values, and even auto-detects and corrects color inversion (e.g. for a photo of a digit drawn in dark ink on white paper, vs. MNIST's native light-on-dark convention).

### Interactive notebook

For a guided, cell-by-cell walkthrough of the entire pipeline (data exploration, training, evaluation, and visual predictions), open:

```
notebooks/exploration.ipynb
```

## Generated Outputs

After running `main.py` or `src/train.py` / `src/evaluate.py`, the following files are produced:

| File | Description |
|---|---|
| `models/handwritten_model.keras` | The trained Keras model |
| `plots/accuracy_plot.png` | Training vs. validation accuracy over epochs |
| `plots/loss_plot.png` | Training vs. validation loss over epochs |
| `plots/confusion_matrix.png` | Confusion matrix heatmap on the test set |
| `plots/classification_report.txt` | Precision, recall, F1-score per digit class |

## Recorded results

The following classification report is present in the project documentation. It records 99.30% accuracy on the 10,000-image MNIST test set; regenerate it with the current training/evaluation code before treating it as a newly reproduced result.

**Classification Report (Recorded Test Run):**

```
              precision    recall  f1-score   support

           0     0.9949    0.9949    0.9949       980
           1     0.9947    0.9956    0.9952      1135
           2     0.9875    0.9981    0.9928      1032
           3     0.9911    0.9950    0.9931      1010
           4     0.9939    0.9990    0.9964       982
           5     0.9888    0.9910    0.9899       892
           6     0.9948    0.9948    0.9948       958
           7     0.9941    0.9883    0.9912      1028
           8     0.9938    0.9928    0.9933       974
           9     0.9960    0.9802    0.9880      1009

    accuracy                         0.9930     10000
   macro avg     0.9930    0.9930    0.9930     10000
weighted avg     0.9930    0.9930    0.9930     10000
```

The confusion matrix shows strong diagonal dominance, with the most common (still rare) confusions occurring between visually similar digits such as 4/9 and 3/5.

## Tech Stack

- **TensorFlow / Keras** — model definition, training, and inference
- **NumPy** — array manipulation and preprocessing
- **Matplotlib / Seaborn** — plotting training curves and the confusion matrix
- **scikit-learn** — classification report and confusion matrix computation
- **Pillow (PIL)** — loading and preprocessing arbitrary input images for prediction

## Notes on Reproducibility

Training uses:

- validation_split = 0.1
- batch_size = 128
- fixed random seeds (TensorFlow and NumPy set to 42)

These settings improve reproducibility, although minor differences between runs may still occur depending on hardware and TensorFlow internals. The documented run reports 99.30% test accuracy. Actual results can vary with the software environment and training run.

## License

This project is intended for educational purposes.