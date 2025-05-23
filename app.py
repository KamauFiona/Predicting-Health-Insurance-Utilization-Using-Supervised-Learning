# streamlit_app.py - Health Insurance Utilization Predictor
import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and encoders
model = joblib.load('insurance_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit UI
st.title("Health Insurance Utilization Predictor")
st.write("Enter the details below to predict if a person is likely to have health insurance.")

# Input fields (all columns used during training)
age = st.selectbox("Age", label_encoders['Age'].classes_)
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
marital_status = st.selectbox("Marital Status", label_encoders['Marital_Status'].classes_)
children = st.slider("Number of Children", 0, 10, 0)
employment = st.selectbox("Employment Status", label_encoders['Employment_Status'].classes_)
income = st.selectbox("Monthly Income", label_encoders['Monthly_Income'].classes_)
had_insurance_last_visit = st.selectbox("Had Insurance Last Visit", label_encoders['Had_Insurance_Last_Visit'].classes_)
had_cancer_screening = st.selectbox("Had Cancer Screening", label_encoders['Had_Cancer_Screening'].classes_)

# Encode input data
input_dict = {
    'Age': label_encoders['Age'].transform([age])[0],
    'Gender': label_encoders['Gender'].transform([gender])[0],
    'Marital_Status': label_encoders['Marital_Status'].transform([marital_status])[0],
    'Children': children,
    'Employment_Status': label_encoders['Employment_Status'].transform([employment])[0],
    'Monthly_Income': label_encoders['Monthly_Income'].transform([income])[0],
    'Had_Insurance_Last_Visit': label_encoders['Had_Insurance_Last_Visit'].transform([had_insurance_last_visit])[0],
    'Had_Cancer_Screening': label_encoders['Had_Cancer_Screening'].transform([had_cancer_screening])[0]
}

# Create DataFrame and reorder columns to match training
input_df = pd.DataFrame([input_dict])
expected_order = ['Age', 'Gender', 'Marital_Status', 'Children', 'Employment_Status',
                  'Monthly_Income', 'Had_Insurance_Last_Visit', 'Had_Cancer_Screening']
input_df = input_df[expected_order]

# Scale input
scaled_input = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    result = "YES" if prediction == 1 else "NO"
    st.success(f"Prediction: The person is likely to HAVE health insurance → {result}")
