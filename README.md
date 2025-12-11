📊 Telecom Customer Churn Prediction – End-to-End ML System

A full-stack machine learning system for predicting customer churn in the telecommunications industry.
This project uses an XGBoost classifier, a FastAPI backend, and a Streamlit frontend to deliver real-time churn prediction, customer risk analysis, and batch scoring capabilities.

🚀 Project Overview

Customer churn is one of the biggest revenue drains in the telecom industry. Early identification of at-risk customers enables companies to apply targeted retention strategies, significantly increasing customer lifetime value.

This project focuses on maximizing recall (79.36%) to ensure the model captures as many potential churners as possible — aligning directly with business impact.

🏗️ System Architecture
┌──────────────────────────────────────────────┐
│                 Streamlit UI                 │
│ • Real-time predictions                      │
│ • Batch scoring                              │
│ • Customer risk analytics                    │
└───────────────────────────────┬──────────────┘
                                │ HTTP/REST
                                ▼
┌──────────────────────────────────────────────┐
│                 FastAPI Backend              │
│ • Prediction endpoint                        │
│ • Preprocessing pipeline                     │
│ • Model + threshold loading                  │
└───────────────────────────────┬──────────────┘
                                │ Load Model
                                ▼
┌──────────────────────────────────────────────┐
│               XGBoost ML Model               │
│ • churn_xgb.pkl                               │
│ • Recall = 79.36% (Primary metric)            │
│ • F1 Score = 0.642                            │
└──────────────────────────────────────────────┘

✨ Key Features
✔ High-Recall ML Model (79.36%)

Captures maximum churners → critical for retention strategy.

✔ FastAPI-Powered REST API

Production-ready inference endpoint deployed on Render.

✔ Interactive Streamlit Frontend

Simple and intuitive interface for non-technical business users.

✔ Real-Time + Batch Predictions

Predict churn for individual customers or entire datasets.

✔ Fully Deployed

Backend: Render

Frontend: Streamlit Cloud

📂 Dataset

Source: Telco Customer Churn Dataset (Kaggle)
File: WA_Fn-UseC_-Telco-Customer-Churn.csv
Shape: 7,043 rows × 21 features

Target Variable

Churn → Yes/No

Imbalance: 73.5% non-churn / 26.5% churn

Feature Categories
Demographics

Gender

SeniorCitizen

Partner

Dependents

Services

PhoneService

MultipleLines

InternetService (DSL / Fiber Optic / No)

OnlineSecurity

OnlineBackup

DeviceProtection

TechSupport

StreamingTV

StreamingMovies

Account Information

Tenure

Contract

PaperlessBilling

PaymentMethod

MonthlyCharges

TotalCharges

🔧 Data Preprocessing
✔ Cleaning

Dropped customerID

Converted TotalCharges to numeric

Filled missing values using median

Removed OnlineBackup due to quality issues

✔ Encoding

Label Encoding for binary fields
One-Hot Encoding for multi-class fields

Final Feature Count: 22 engineered features
🤖 Model Development
Train/Test Split

Train: 80%

Test: 20%

random_state = 42

🟩 Primary Model — XGBoost Classifier
Hyperparameters
n_estimators=200
learning_rate=0.05
max_depth=5
subsample=0.8
colsample_bytree=0.8
scale_pos_weight=2.7

Performance
Metric	Score
Accuracy	76.58%
F1 Score	0.642
⭐ Recall	79.36%
Why Selected?

Highest recall → captures 24% more churners than Logistic Regression

Balanced F1 score

Handles class imbalance effectively

🟨 Secondary Model — Logistic Regression
Metric	Score
Accuracy	81.83%
Recall	55.50%
F1 Score	0.618

Good for interpretability, but not suitable for high-recall objectives.

🎯 Business Metric Prioritization

Recall → Most important (don’t miss churners)

F1 Score → Balanced evaluation

Accuracy → Least important (misleading on imbalanced datasets)

Example:

A stupid model predicting “No Churn” for everyone gets 73.5% accuracy → completely useless.

⚙️ Threshold Optimization

The default 0.5 threshold is not suitable for imbalanced churn data.

After optimization:
Optimal threshold = 0.01


This aggressively maximizes recall for business impact.

📦 Saved Artifacts
churn_xgb.pkl      # Trained XGBoost model
threshold.pkl      # Optimized threshold = 0.01

🌐 FastAPI Backend

Example FastAPI code snippet:

from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/churn_xgb.pkl")
threshold = joblib.load("models/threshold.pkl")

@app.post("/predict")
def predict(data: dict):
    features = preprocess(data)
    prob = model.predict_proba([features])[0][1]
    pred = int(prob >= threshold)

    return {
        "churn_probability": prob,
        "prediction": pred,
        "risk_level": "High" if pred == 1 else "Low"
    }


Deployed at:

https://churn-prediction-2qrp.onrender.com/

🖥️ Streamlit Frontend

Real-time prediction

Batch CSV upload

Visual analytics

Deployed at:

https://churn-frontend-g3ku8j45b7mfsg6s4ztjfy.streamlit.app/

🛠️ Installation & Setup
Clone repository
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction

Backend Setup
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

Frontend Setup
cd churn-frontend
pip install -r requirements.txt
streamlit run app.py

🧪 API Usage Example
curl -X POST "https://churn-prediction-2qrp.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 12,
    "MonthlyCharges": 70
}'

📈 Business Impact

Assumptions:

Avg revenue/user = $64/month

Lifetime value ≈ $1500

Retention offer cost: $75

Retention success: 40%

With XGBoost (79.36% recall)

Correctly identifies 1,483 churners

Potential revenue saved: $890,000

Campaign cost: $111,225

Net Benefit = $778,775 saved

With Logistic Regression (55.50% recall)

Saves only $623,000

👉 XGBoost prevents $267,000 additional revenue loss.

📌 Repository Structure
telecom-churn-prediction/
│
├── backend/
│   ├── app.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── churn_xgb.pkl
│   │   └── threshold.pkl
│   ├── requirements.txt
│   └── render.yaml
│
├── churn-frontend/
│   ├── app.py
│   ├── pages/
│   │   ├── 01_prediction.py
│   │   ├── 02_batch.py
│   │   └── 03_analytics.py
│   ├── utils/
│   │   ├── api_client.py
│   │   └── visualizations.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
│
├── notebooks/
│   └── main_analysis.ipynb
│
├── README.md
└── .gitignore

🔮 Future Enhancements
ML Improvements

SHAP explainability

Ensemble models (XGBoost + LightGBM)

AutoML hyperparameter search

Time-series behavior modeling

System Enhancements

Model drift monitoring

Automated retraining workflow

CRM integration

Alert system for high-risk customers

📬 Contact

For issues, suggestions, or collaboration:
📧 workwithanshuman9468@gmail.com

✅ Project Status

Production Ready

Model Version: 1.0

Last Updated: Dec 2025
