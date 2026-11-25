# Breast Cancer Classification with Logistic Regression

A clean, reproducible machine learning pipeline built with `scikit-learn` to classify breast tumor samples as **benign (`2`)** or **malignant (`4`)** using Logistic Regression.

## Project Structure

```text
cancer-classification-ml/
├── data/
│   ├── cancer_data.csv          # Cleaned dataset (683 samples, 9 features + ID & Class)
│   └── new_datasets.csv         # Sample unlabeled feature data for inference
├── models/
│   └── logistic_model.pkl       # Trained Logistic Regression model artifact
├── notebooks/
│   └── cancer_detection_log_reg.ipynb  # Cleaned exploratory Jupyter Notebook
├── src/
│   ├── train.py                 # Trains, evaluates, and saves the model
│   └── predict.py               # Loads the saved model and exports predictions
├── .gitignore
├── requirements.txt
└── README.md
```

## Model Performance
- **Test Set Accuracy:** 95.62%
- **10-Fold Cross-Validation Accuracy:** 96.70% (±1.97%)
- **Confusion Matrix (Test Set):** `[[84, 3], [3, 47]]`

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cancer-classification-ml.git
   cd cancer-classification-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train and save the model:**
   ```bash
   python src/train.py
   ```
   Trains the classifier on `data/cancer_data.csv` and saves the model to `models/logistic_model.pkl`.

2. **Generate predictions on new data:**
   ```bash
   python src/predict.py
   ```
   Reads `data/new_datasets.csv` (9 feature columns) and saves predictions to `data/new_dataset_with_predictions.csv`.
