import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Multi-Disease Predictor", layout="wide")

# Load models
@st.cache_resource
def load_models():
    diabetes_model = joblib.load('diabetes_model.pkl')
    heart_model = joblib.load('heart_disease_model.pkl')
    kidney_model = joblib.load('kidney_disease_model.pkl')
    return diabetes_model, heart_model, kidney_model

diabetes_model, heart_model, kidney_model = load_models()

# Title
st.title("🏥 AI-Powered Multi-Disease Prediction System")
st.markdown("**Developed by:** Srustisri Panda, Puja Rani Mishra, Kajal Roul")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
disease = st.sidebar.radio("Select Disease to Predict:", ["Diabetes", "Heart Disease", "Kidney Disease", "Full Health Check"])

# Diabetes Prediction
if disease == "Diabetes" or disease == "Full Health Check":
    st.subheader("🔬 Diabetes Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    with col2:
        bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col3:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=100)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
    with col4:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    if st.button("Predict Diabetes Risk"):
        diabetes_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        diabetes_pred = diabetes_model.predict(diabetes_input)
        diabetes_prob = diabetes_model.predict_proba(diabetes_input)
        
        if diabetes_pred[0] == 1:
            st.warning(f"⚠️ **Diabetes Risk: POSITIVE** (Confidence: {diabetes_prob[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ **Diabetes Risk: NEGATIVE** (Confidence: {diabetes_prob[0][0]*100:.2f}%)")

# Heart Disease Prediction
if disease == "Heart Disease" or disease == "Full Health Check":
    st.subheader("❤️ Heart Disease Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age_h = st.number_input("Age (Heart)", min_value=0, max_value=120, value=50, key="age_h")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    with col2:
        cp = st.number_input("Chest Pain Type", min_value=0, max_value=3, value=1)
        trestbps = st.number_input("Resting BP", min_value=0, max_value=250, value=130)
    with col3:
        chol = st.number_input("Cholesterol", min_value=0, max_value=400, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col4:
        restecg = st.number_input("Resting ECG", min_value=0, max_value=2, value=0)
        thalach = st.number_input("Max Heart Rate", min_value=0, max_value=250, value=150)
    
    exang = st.number_input("Exercise Induced Angina", min_value=0, max_value=1, value=0)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("ST Slope", min_value=0, max_value=2, value=1)
    ca = st.number_input("Major Vessels", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, value=2)
    
    if st.button("Predict Heart Disease Risk"):
        heart_input = np.array([[age_h, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_pred = heart_model.predict(heart_input)
        
        if heart_pred[0] == 1:
            st.warning("⚠️ **Heart Disease Risk: POSITIVE**")
        else:
            st.success("✅ **Heart Disease Risk: NEGATIVE**")

# Kidney Disease Prediction
if disease == "Kidney Disease" or disease == "Full Health Check":
    st.subheader("🫘 Kidney Disease Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        age_k = st.number_input("Age (Kidney)", min_value=0, max_value=120, value=50, key="age_k")
        bp_k = st.number_input("Blood Pressure (Kidney)", min_value=0, max_value=250, value=80, key="bp_k")
    with col2:
        sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.03, value=1.02, step=0.001)
        al = st.number_input("Albumin", min_value=0, max_value=5, value=0)
    with col3:
        su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
    
    if st.button("Predict Kidney Disease Risk"):
        kidney_input = np.array([[age_k, bp_k, sg, al, su]])
        kidney_pred = kidney_model.predict(kidney_input)
        
        if kidney_pred[0] == 1:
            st.warning("⚠️ **Kidney Disease Risk: POSITIVE**")
        else:
            st.success("✅ **Kidney Disease Risk: NEGATIVE**")

st.markdown("---")
st.markdown("**Disclaimer:** This system is for informational purposes only and should not replace professional medical advice.")
