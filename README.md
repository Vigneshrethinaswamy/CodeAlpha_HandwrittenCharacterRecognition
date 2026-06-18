# CodeAlpha — Handwritten Character Recognition

A complete, end-to-end **Handwritten Digit Recognition** system built with **TensorFlow/Keras**, trained on the classic **MNIST** dataset. The project covers data loading and preprocessing, CNN model design, training, evaluation, visualization, and single-image inference — all organized into a clean, modular, production-style codebase.

## Overview

This project trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from 28×28 grayscale images. After 10 epochs of training, the model achieves **~99.1% accuracy** on the held-out MNIST test set.

| Metric | Value |
|---|---|
| Test Accuracy | **99.30%** |
| Test Loss | 0.0217 |
| Training Epochs | 15 (with EarlyStopping) |
| Total Parameters | 225,034 |

## Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
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
cd CodeAlpha_HandwrittenCharacterRecognition
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
3. Train for 10 epochs (with a 10% validation split).
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

## Results

**Classification Report (Test Set):**

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

These settings improve reproducibility, although minor differences between runs may still occur depending on hardware and TensorFlow internals. The model consistently achieves accuracy in the high 98%–99% range on the MNIST test set.

## License

This project was built as part of the **CodeAlpha** internship/training program and is intended for educational purposes.