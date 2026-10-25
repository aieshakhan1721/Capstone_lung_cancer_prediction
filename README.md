# Lung Cancer Prediction Using Machine Learning

## Overview

This project explores the use of machine learning classification algorithms to predict the lung cancer survey label using the features available in a survey dataset.

Six classification models are trained and evaluated using a common train-test split. Their performance is compared using accuracy, precision, recall, and F1-score. The project also includes confusion matrices, a feature correlation heatmap, and feature distribution visualizations.

**Note:** This project is for educational purposes only. Its predictions are not a medical diagnosis or a substitute for professional medical advice.

## Objectives

- Explore and preprocess a lung cancer survey dataset.
- Separate input features from the target variable.
- Train and compare six machine learning classifiers.
- Evaluate models using multiple classification metrics.
- Visualize model performance and feature relationships.

## Algorithms Used

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Support Vector Machine (SVM)
5. Gaussian Naive Bayes
6. Random Forest

## Technologies and Libraries

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses `survey_lung_cancer.csv`.

The dataset contains survey features and a target column named `LUNG_CANCER`.

Place the CSV file in the project directory before running the script. 

## Project Structure

lung-cancer-prediction/
│
├── survey_lung_cancer.csv
├── lung_cancer_prediction.py
├── README.md
└── requirements.txt

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd lung-cancer-prediction
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the libraries directly:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run

Run the Python script from the project directory:

```bash
python lung_cancer_prediction.py
```

The script trains the models, prints evaluation metrics, and displays the comparison charts and heatmaps.

## Evaluation Metrics

- **Accuracy:** Proportion of correctly classified test samples.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases identified correctly.
- **F1-score:** Harmonic mean of precision and recall.

The positive class is defined as `YES` when the dataset uses `YES` and `NO` target labels.

## Visualizations

The project generates:

- A bar chart comparing model accuracy.
- Confusion matrices for the six classifiers.
- A feature correlation heatmap.
- Histograms for numeric features.
- Count plots for categorical features.

## Results

The script prints the evaluation metrics for each model and creates a comparison table.

Record the actual results obtained when you run the code. Do not assume that a particular algorithm performs best without comparing the measured results.

## Limitations

- Results depend on the dataset, preprocessing, and train-test split.
- Similar performance metrics do not necessarily mean that models make identical predictions.
- A single train-test split may not provide a reliable estimate of generalization performance.
- Survey data may contain bias, class imbalance, and limitations in the recorded features.
- This model is intended for learning and experimentation, not clinical diagnosis.

## Future Improvements

- Use stratified cross-validation for more robust evaluation.
- Add hyperparameter tuning using GridSearchCV.
- Improve preprocessing with leakage-safe pipelines.
- Examine feature importance and model interpretability.
- Evaluate class imbalance and compare with a majority-class baseline.

## App url: https://capstonelungcancerprediction-lkasp5vwpqkcx3vrfkahzy.streamlit.app/
## Author
Ayesha Shaik
