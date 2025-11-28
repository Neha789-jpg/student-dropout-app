import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# App Title
# ----------------------
st.title("Student Dropout Prediction")
st.write("Upload a CSV file containing student data to predict dropouts.")

# ----------------------
# Load trained model

clf = joblib.load("model.pkl")
label_mapping = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

# ----------------------
# Columns used during training
# ----------------------
feature_names = ["Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]  

# ----------------------
# File upload
# ----------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    # ----------------------
    # Check for missing columns
    # ----------------------
    missing_features = set(feature_names) - set(new_data.columns)
    if missing_features:
        st.error(f"The following required features are missing: {missing_features}")
    else:
        
        X = new_data[feature_names]

        # ----------------------
        # Make predictions
        # ----------------------
        predictions = clf.predict(X)

        predictions_named = [label_mapping[p] for p in predictions]

        # Display predictions
        new_data['Prediction'] = predictions_named
        st.write("Predictions for uploaded data:")
        st.dataframe(new_data)
