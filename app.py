import streamlit as st
import pandas as pd
#import joblib
from joblib import load
from pathlib import Path

# Load trained model
#model = load("Loan_model.joblib")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model_Dir" / "Loan_Model.joblib"

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Loan Status Prediction", page_icon="🏦")

st.title("🏦 Loan Status Prediction")
st.write("Enter the applicant details below to predict loan approval.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Amount Term (months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Encode inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0

dep_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}
dependents = dep_map[dependents]

education = 0 if education == "Graduate" else 1

property_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}
property_area = property_map[property_area]

# Create dataframe
input_data = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

# Convert categorical variables into dummy variables
input_data = pd.get_dummies(
    input_data,
    columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ],
    drop_first=True
)

# Get the feature names used during model training
expected_features = model.feature_names_in_

# Add missing columns and remove extra columns
input_data = input_data.reindex(
    columns=expected_features,
    fill_value=0
)


# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")