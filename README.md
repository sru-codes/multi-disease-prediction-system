# AI-Powered Multi-Disease Prediction System (Major Project)

**Author:** Srustisri Panda  
**Topic:** Healthcare & Medical AI  
**Project ID:** Major Project #1 (from AI/ML Major Project List)

---

## 📌 Project Overview
The **AI-Powered Multi-Disease Prediction System** is a sophisticated healthcare solution that allows users to predict the likelihood of multiple diseases simultaneously. By analyzing key health parameters such as glucose levels, blood pressure, BMI, and age, the system provides instant risk assessments for **Diabetes**, **Heart Disease**, and **Kidney Disease**.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Algorithms:**
  - **Diabetes:** Random Forest Classifier
  - **Heart Disease:** Logistic Regression
  - **Kidney Disease:** Support Vector Machine (SVM)

## 📊 Features & Functionality
- **Multi-Model Integration:** Features three specialized machine learning models trained on distinct medical datasets.
- **Unified Prediction Engine:** A single interface (`predict.py`) that processes user vitals and generates a comprehensive health report.
- **Robust Preprocessing:** Handles data scaling and encoding to ensure high prediction accuracy.
- **Explainable AI:** Provides clear positive/negative results based on established medical data patterns.

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install pandas scikit-learn joblib
```

### Running the Project
1. **Train All Models:**
   ```bash
   python train.py
   ```
2. **Generate Predictions:**
   ```bash
   python predict.py
   ```
   You can modify the sample inputs in `predict.py` to test different health scenarios.

---
*Developed as part of the AI/ML with Python Course.*
