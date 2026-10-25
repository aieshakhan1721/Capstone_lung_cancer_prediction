
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Lung Cancer Survey Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Lung Cancer Survey Prediction")
st.write(
    "Enter the survey details below to get a prediction "
    "from the trained Logistic Regression model."
)

st.warning(
    "Educational project only. This model is not a medical "
    "diagnostic tool and must not be used to diagnose or "
    "rule out lung cancer."
)

# --------------------------------------------------
# 2. Load and clean the dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("survey_lung_cancer.csv")

    # Remove leading/trailing spaces from column names
    data.columns = data.columns.str.strip()

    return data


try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "survey_lung_cancer.csv was not found. "
        "Place it in the project folder."
    )
    st.stop()

# --------------------------------------------------
# 3. Load the saved Logistic Regression model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("lung_cancer_model.joblib")


try:
    artifact = load_model()
except FileNotFoundError:
    st.error(
        "Trained model not found. Run train_model.py first "
        "to create lung_cancer_logistic_model.joblib."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

model = artifact["model"]
feature_columns = artifact["feature_columns"]
categorical_cols = artifact["categorical_columns"]
numeric_cols = artifact["numeric_columns"]
categories = artifact["categories"]
numeric_ranges = artifact["numeric_ranges"]
target_classes = artifact.get("target_classes", [])

# --------------------------------------------------
# 4. Validate dataset columns against the saved model
# --------------------------------------------------
missing_columns = [
    col for col in feature_columns
    if col not in data.columns
]

if missing_columns:
    st.error(
        "The CSV file does not contain all columns expected "
        "by the trained model."
    )
    st.write("Missing columns:", missing_columns)
    st.write("Available CSV columns:", data.columns.tolist())
    st.stop()

# --------------------------------------------------
# 5. Create the input form
# --------------------------------------------------
st.subheader("Enter Survey Details")

with st.form("prediction_form"):

    user_input = {}

    for col in feature_columns:

        # Categorical features
        if col in categorical_cols:

            options = categories.get(col, [])

            if options:
                user_input[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options=options,
                    format_func=lambda value: str(value)
                )
            else:
                user_input[col] = st.text_input(
                    col.replace("_", " ").title()
                )

        # Numerical features
        elif col in numeric_cols:

            min_value, max_value, median_value = (
                numeric_ranges[col]
            )

            # Avoid an invalid number_input range
            if min_value == max_value:
                user_input[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(median_value)
                )
            else:
                user_input[col] = st.number_input(
                    col.replace("_", " ").title(),
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(median_value)
                )

    submitted = st.form_submit_button("Predict")

# --------------------------------------------------
# 6. Make a prediction
# --------------------------------------------------
if submitted:

    # Build input DataFrame in the exact feature order
    input_df = pd.DataFrame(
        [user_input],
        columns=feature_columns
    )

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if str(prediction).strip().upper() == "YES":
            st.error(
                "Model prediction: YES (positive survey label)"
            )
        elif str(prediction).strip().upper() == "NO":
            st.success(
                "Model prediction: NO (negative survey label)"
            )
        else:
            st.info(f"Model prediction: {prediction}")

        # Show prediction probabilities when available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Model Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability (%)": (
                    probabilities * 100
                ).round(2)
            })

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These are model probabilities for the survey "
                "labels, not the probability that a person "
                "has lung cancer."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# 7. Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Built with Python, Scikit-learn, Pandas and Streamlit."
)
