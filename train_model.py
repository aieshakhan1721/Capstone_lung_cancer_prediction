
# 1. Import required libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 2. Load the dataset
data = pd.read_csv("survey_lung_cancer.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# 3. Check whether the target column exists
if "LUNG_CANCER" not in data.columns:
    raise ValueError("LUNG_CANCER column not found in the dataset.")

# Remove rows where the target is missing
data = data.dropna(subset=["LUNG_CANCER"]).copy()

# Standardize target labels
data["LUNG_CANCER"] = (
    data["LUNG_CANCER"].astype(str).str.strip().str.upper()
)

# 4. Separate input features and target
X = data.drop(columns=["LUNG_CANCER"])
y = data["LUNG_CANCER"]

# 5. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

# 6. Create the numerical preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Create the categorical preprocessing pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 9. Create the Logistic Regression model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 10. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1/3,
    random_state=0,
    stratify=y
)

# 11. Train the model using training data
model.fit(X_train, y_train)

# 12. Make predictions on testing data
y_pred = model.predict(X_test)

# 13. Evaluate model performance
print("\nLOGISTIC REGRESSION RESULTS")
print("----------------------------")

print("Accuracy:", round(
    accuracy_score(y_test, y_pred), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 14. Retrain the final model on all available data
# Do this only after completing the evaluation above.
model.fit(X, y)

# 15. Save the trained model and metadata
artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
    "categories": {
        col: X[col].dropna().unique().tolist()
        for col in categorical_cols
    },
    "numeric_ranges": {
        col: [
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].median())
        ]
        for col in numeric_cols
    },
    "target_classes": sorted(y.unique().tolist())
}

joblib.dump(
    artifact,
    "lung_cancer_logistic_model.joblib"
)

print("\nModel saved successfully!")
print("File: lung_cancer_logistic_model.joblib")
