import streamlit as st
import requests

st.title("📊 Customer Churn Prediction App")

API_URL = "https://churn-prediction-2qrp.onrender.com/predict"

st.write("Enter customer features to predict churn:")

# EXAMPLE: 5 features, change to match your dataset!
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")
f5 = st.number_input("Feature 5")

features = [f1, f2, f3, f4, f5]

if st.button("Predict"):
    payload = {"features": features}
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Churn Probability: {result['probability']:.2f}")
        st.info(f"Prediction: {'Churn' if result['prediction']==1 else 'Not Churn'}")
    else:
        st.error("Error connecting to backend API.")
