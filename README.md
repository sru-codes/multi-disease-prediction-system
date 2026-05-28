# AI-Powered Multi-Disease Prediction System

This is a major project that provides predictions for multiple diseases (Diabetes, Heart Disease, Kidney Disease) based on user health parameters.

## Features
- Separate Machine Learning models for each disease.
- Unified prediction engine for a comprehensive health check.
- Explainable results based on health vitals.

## Models Used
- **Diabetes**: Random Forest Classifier
- **Heart Disease**: Logistic Regression
- **Kidney Disease**: Support Vector Classifier (SVC)

## How to run
1. Install dependencies: `pip install pandas scikit-learn joblib`
2. Run training script: `python train.py`
3. Run prediction script: `python predict.py`
