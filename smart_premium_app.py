import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="SmartPremium",
    page_icon="💰",
    layout="wide"
)

# -----------------------------------
# Load Model
# -----------------------------------
MODEL_PATH = os.path.join("models", "best_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------------
# Header
# -----------------------------------
st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#1E88E5;'>💰 SmartPremium</h1>
    <h4>Insurance Premium Prediction System</h4>
    <p style='color:gray;'>
        Predict customer insurance premium using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# User Inputs
# -----------------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=1
    )

    education = st.selectbox(
        "Education Level",
        [
            "High School",
            "Bachelor's",
            "Master's",
            "PhD"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Employed",
            "Self-Employed",
            "Unemployed"
        ]
    )

    health_score = st.slider(
        "Health Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

with col2:

    location = st.selectbox(
        "Location",
        [
            "Urban",
            "Suburban",
            "Rural"
        ]
    )

    policy_type = st.selectbox(
        "Policy Type",
        [
            "Basic",
            "Comprehensive",
            "Premium"
        ]
    )

    previous_claims = st.number_input(
        "Previous Claims",
        min_value=0,
        value=0
    )

    vehicle_age = st.number_input(
        "Vehicle Age",
        min_value=0,
        value=3
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

    insurance_duration = st.number_input(
        "Insurance Duration (Years)",
        min_value=1,
        max_value=30,
        value=5
    )

    policy_start_date = st.date_input(
        "Policy Start Date"
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "Yes",
            "No"
        ]
    )

    exercise = st.selectbox(
        "Exercise Frequency",
        [
            "Daily",
            "Weekly",
            "Monthly",
            "Rarely"
        ]
    )

    property_type = st.selectbox(
        "Property Type",
        [
            "House",
            "Apartment",
            "Condo"
        ]
    )

# -----------------------------------
# Predict
# -----------------------------------

if st.button("Predict Premium"):

    input_df = pd.DataFrame({

        "Age": [age],
        "Gender": [gender],
        "Annual Income": [annual_income],
        "Marital Status": [marital_status],
        "Number of Dependents": [dependents],
        "Education Level": [education],
        "Occupation": [occupation],
        "Health Score": [health_score],
        "Location": [location],
        "Policy Type": [policy_type],
        "Previous Claims": [previous_claims],
        "Vehicle Age": [vehicle_age],
        "Credit Score": [credit_score],
        "Insurance Duration": [insurance_duration],
        "Smoking Status": [smoking_status],
        "Exercise Frequency": [exercise],
        "Property Type": [property_type],

        # Required by trained model
        "Policy_Year": [policy_start_date.year],
        "Policy_Month": [policy_start_date.month],
        "Policy_Day": [policy_start_date.day]

    })

    try:

        prediction = model.predict(input_df)[0]

        st.success(
            f"💰 Predicted Insurance Premium: ₹ {prediction:,.2f}"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")