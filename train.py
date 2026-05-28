import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. Diabetes Prediction Model
def train_diabetes():
    data = {
        'Pregnancies': [1, 8, 1, 1, 0, 5, 3, 10, 2, 8],
        'Glucose': [85, 183, 89, 137, 116, 110, 78, 115, 197, 125],
        'BloodPressure': [66, 64, 66, 40, 74, 50, 50, 0, 70, 96],
        'SkinThickness': [29, 0, 23, 35, 0, 0, 32, 0, 45, 0],
        'Insulin': [0, 0, 94, 168, 0, 0, 88, 0, 543, 0],
        'BMI': [26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 31.0, 35.3, 30.5, 0.0],
        'DiabetesPedigreeFunction': [0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.248, 0.134, 0.158, 0.232],
        'Age': [31, 32, 21, 33, 30, 26, 26, 29, 53, 54],
        'Outcome': [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'diabetes_model.pkl')
    print("Diabetes model trained.")

# 2. Heart Disease Prediction Model
def train_heart():
    data = {
        'age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
        'sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        'cp': [3, 2, 1, 1, 0, 0, 1, 1, 2, 2],
        'trestbps': [145, 130, 130, 120, 120, 140, 140, 120, 172, 150],
        'chol': [233, 250, 204, 236, 354, 192, 294, 263, 199, 168],
        'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'restecg': [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        'thalach': [150, 187, 172, 178, 163, 148, 153, 173, 162, 174],
        'exang': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 0.4, 1.3, 0.0, 0.5, 1.6],
        'slope': [0, 0, 2, 2, 2, 1, 1, 2, 1, 2],
        'ca': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'thal': [1, 2, 2, 2, 2, 1, 2, 3, 3, 2],
        'target': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, 'heart_disease_model.pkl')
    print("Heart disease model trained.")

# 3. Kidney Disease Prediction Model
def train_kidney():
    data = {
        'age': [48, 7, 62, 48, 51, 60, 68, 24, 52, 53],
        'bp': [80, 50, 80, 70, 80, 90, 70, 80, 100, 90],
        'sg': [1.020, 1.020, 1.010, 1.005, 1.010, 1.015, 1.010, 1.015, 1.015, 1.020],
        'al': [1, 4, 2, 4, 2, 3, 0, 2, 3, 2],
        'su': [0, 0, 3, 0, 0, 0, 0, 4, 0, 0],
        'target': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']
    model = SVC()
    model.fit(X, y)
    joblib.dump(model, 'kidney_disease_model.pkl')
    print("Kidney disease model trained.")

if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_kidney()
    print("All models trained and saved successfully!")
