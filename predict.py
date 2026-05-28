import joblib
import numpy as np

def predict_all(diabetes_input, heart_input, kidney_input):
    results = {}
    
    # Diabetes
    try:
        diabetes_model = joblib.load('diabetes_model.pkl')
        diabetes_pred = diabetes_model.predict([diabetes_input])
        results['Diabetes'] = 'Positive' if diabetes_pred[0] == 1 else 'Negative'
    except Exception as e:
        results['Diabetes'] = f"Error: {e}"

    # Heart Disease
    try:
        heart_model = joblib.load('heart_disease_model.pkl')
        heart_pred = heart_model.predict([heart_input])
        results['Heart Disease'] = 'Positive' if heart_pred[0] == 1 else 'Negative'
    except Exception as e:
        results['Heart Disease'] = f"Error: {e}"

    # Kidney Disease
    try:
        kidney_model = joblib.load('kidney_disease_model.pkl')
        kidney_pred = kidney_model.predict([kidney_input])
        results['Kidney Disease'] = 'Positive' if kidney_pred[0] == 1 else 'Negative'
    except Exception as e:
        results['Kidney Disease'] = f"Error: {e}"

    return results

if __name__ == "__main__":
    # Sample inputs
    diabetes_sample = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    heart_sample = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    kidney_sample = [48, 80, 1.020, 1, 0]
    
    print("Running Multi-Disease Prediction...")
    predictions = predict_all(diabetes_sample, heart_sample, kidney_sample)
    for disease, result in predictions.items():
        print(f"{disease}: {result}")
