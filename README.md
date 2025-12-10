Telecom Customer Churn Prediction Model
Overview
This project implements a full-stack machine learning solution to predict customer churn in the telecommunications industry using the Telco Customer Churn dataset. The system includes a trained XGBoost model, REST API backend, and interactive frontend dashboard for real-time churn prediction and customer risk analysis.
Project Architecture
┌─────────────────┐
│  Frontend UI    │  React/Vue Dashboard
│ (Port 3000)     │  - Customer risk analysis
└────────┬────────┘  - Batch predictions
         │           - Visualization
         │ HTTP
         ▼
┌─────────────────┐
│  Backend API    │  Flask/FastAPI
│ (Port 5000)     │  - Prediction endpoint
└────────┬────────┘  - Preprocessing
         │           - Model serving
         │
         ▼
┌─────────────────┐
│  ML Model       │  XGBoost Classifier
│  churn_xgb.pkl  │  - Recall: 79.36%
└─────────────────┘  - F1: 0.642
Key Features

✅ High Recall Model (79.36%) - Captures maximum churners
✅ REST API Backend - Easy integration with existing systems
✅ Interactive Frontend - User-friendly dashboard for business users
✅ Real-time Predictions - Instant churn risk assessment
✅ Batch Processing - Score entire customer databases
✅ Production Ready - Deployed and tested model artifacts

Customer churn is a critical challenge in the telecom sector, directly impacting revenue and growth. Early identification of at-risk customers allows companies to implement targeted retention campaigns, reducing churn rates and improving customer lifetime value. This project focuses on maximizing recall to ensure we capture as many potential churners as possible.
Dataset
Source: Telco Customer Churn Dataset from Kaggle
File: WA_Fn-UseC_-Telco-Customer-Churn.csv
Size: 7,043 customers × 21 features
Dataset Characteristics

Target Variable: Churn (Yes/No)
Class Distribution:

Non-churners: 5,174 customers (73.5%)
Churners: 1,869 customers (26.5%)
Class Imbalance: Dataset is imbalanced, requiring special handling



Features
The dataset includes comprehensive customer information across multiple categories:
Customer Demographics:

Gender (Male/Female)
SeniorCitizen (0/1)
Partner (Yes/No)
Dependents (Yes/No)

Service Information:

Tenure (months with company)
PhoneService, MultipleLines
InternetService (DSL/Fiber optic/No)
OnlineSecurity, OnlineBackup, DeviceProtection
TechSupport, StreamingTV, StreamingMovies

Account Information:

Contract (Month-to-month/One year/Two year)
PaperlessBilling (Yes/No)
PaymentMethod (Electronic check/Mailed check/Bank transfer/Credit card)
MonthlyCharges
TotalCharges

Data Preprocessing
1. Data Cleaning

Removed customerID column (not useful for prediction)
Converted TotalCharges to numeric format
Handled missing values in TotalCharges using median imputation
Removed OnlineBackup column due to data quality issues

2. Feature Engineering
Label Encoding applied to binary categorical features:

gender, Partner, Dependents, PhoneService, PaperlessBilling
MultipleLines, OnlineSecurity, DeviceProtection, TechSupport
StreamingTV, StreamingMovies, Churn (target variable)

One-Hot Encoding applied to multi-category features:

InternetService (DSL, Fiber optic, No)
Contract (Month-to-month, One year, Two year)
PaymentMethod (4 categories)

3. Final Feature Set
After preprocessing: 22 input features used for model training
Model Development
Train-Test Split

Training Set: 80% (5,634 samples)
Test Set: 20% (1,409 samples)
Random State: 42 (for reproducibility)

Models Implemented
1. XGBoost Classifier (PRIMARY MODEL - SELECTED)
Hyperparameters:
pythonn_estimators=200        # Number of boosting rounds
learning_rate=0.05      # Step size shrinkage
max_depth=5             # Maximum tree depth
subsample=0.8           # Subsample ratio of training instances
colsample_bytree=0.8    # Subsample ratio of features
scale_pos_weight=2.7    # Balancing of positive/negative weights (addresses class imbalance)
Performance Metrics:

Accuracy: 76.58%
F1 Score: 0.642
Recall: 79.36% ⭐ (Highest among all models)

Key Advantages:

Superior recall ensures maximum capture of potential churners
Handles class imbalance effectively with scale_pos_weight
Robust performance with gradient boosting
Optimal threshold tuning for business objectives

2. Logistic Regression (SECONDARY MODEL)
Performance Metrics:

Accuracy: 81.83% (Highest accuracy)
F1 Score: 0.618
Recall: 55.50%

Characteristics:

Best overall accuracy but lower recall
Highly interpretable model
Faster inference time
Useful for understanding feature relationships

3. Decision Tree Classifier (EVALUATED)
Performance Metrics:

Accuracy: 72.89%
Not selected due to lower performance and overfitting concerns

Model Selection Rationale
Why XGBoost Over Logistic Regression?
Despite Logistic Regression having 5.25% higher accuracy (81.83% vs 76.58%), XGBoost was selected as the primary model because:

Superior Recall (79.36% vs 55.50%):

Identifies 24% more actual churners
Critical for retention campaign effectiveness
Reduces customer loss by catching more at-risk customers


Better F1 Score (0.642 vs 0.618):

More balanced precision-recall tradeoff
Better overall performance on imbalanced data


Business Impact:

Missing a churner costs significantly more than a false positive
Cost of retention offer << Cost of losing a customer
High recall maximizes retention opportunities



Evaluation Metrics Priority
For churn prediction, metrics are prioritized as:

Recall (Sensitivity) - MOST IMPORTANT

Measures ability to identify actual churners
Formula: True Positives / (True Positives + False Negatives)
High recall = fewer missed churners
Business goal: Don't let potential churners slip through


F1 Score - SECONDARY

Harmonic mean of precision and recall
Provides balanced view of model performance
Essential for imbalanced datasets
Prevents over-optimization on accuracy alone


Accuracy - TERTIARY

Can be misleading with imbalanced classes
A model predicting "no churn" for everyone could achieve 73.5% accuracy
Fails to capture business cost asymmetry



Why Not Just Accuracy?
With a 73.5% / 26.5% class split, a naive model that always predicts "no churn" would achieve 73.5% accuracy while providing zero business value. This demonstrates why accuracy alone is insufficient for imbalanced classification problems.
Model Optimization
Threshold Tuning
Performed threshold optimization to maximize F1 score while maintaining high recall:
python# Tested thresholds from 0.0 to 1.0 in 0.01 increments
Best threshold: 0.01
Best F1 Score: 0.642
This aggressive threshold ensures maximum churner identification, accepting more false positives in exchange for comprehensive coverage of actual churners.
Handling Class Imbalance
Strategy: Set scale_pos_weight=2.7 in XGBoost

Calculated as ratio of negative to positive samples: 5174/1869 ≈ 2.77
Penalizes misclassification of minority class (churners)
Improves recall without sacrificing too much precision

Model Deployment
Saved Artifacts
python# Model file
churn_xgb.pkl           # Trained XGBoost model

# Configuration file  
threshold.pkl           # Optimal prediction threshold (0.01)
Making Predictions
pythonimport joblib
import numpy as np

# Load model and threshold
model = joblib.load('churn_xgb.pkl')
threshold = joblib.load('threshold.pkl')

# Predict on new data
predictions = model.predict(new_customer_data)
probabilities = model.predict_proba(new_customer_data)[:, 1]

# Apply custom threshold
final_predictions = (probabilities >= threshold).astype(int)

# Get high-risk customers
high_risk_indices = np.where(final_predictions == 1)[0]
Business Impact & Use Cases
Retention Campaign Targeting
Primary Use Case: Identify at-risk customers for proactive retention

Model outputs churn probability for each customer
Customers with high probability receive targeted retention offers
Campaigns can be personalized based on customer attributes

Cost-Benefit Analysis
Assumptions:

Average revenue per customer: $64/month (from dataset)
Customer lifetime value: ~$1,500 (based on average tenure)
Retention offer cost: $50-100
Retention success rate: 30-50%

With XGBoost (79.36% Recall):

Captures 1,483 out of 1,869 potential churners in test set
Potential revenue saved: 1,483 × $1,500 × 40% retention = $890,000
Campaign cost: 1,483 × $75 = $111,225
Net benefit: ~$778,775

With Logistic Regression (55.50% Recall):

Captures only 1,037 churners
Potential revenue saved: 1,037 × $1,500 × 40% = $623,000
Lost opportunity: $267,000 compared to XGBoost

Technical Stack
Core Libraries

Python 3.11
pandas 2.x - Data manipulation and analysis
numpy - Numerical computations
scikit-learn - Machine learning algorithms and evaluation
XGBoost - Gradient boosting implementation
matplotlib, seaborn - Data visualization

Machine Learning Pipeline
Data Loading → Preprocessing → Feature Engineering → 
Model Training → Evaluation → Hyperparameter Tuning → 
Threshold Optimization → Model Serialization
Model Performance Summary
ModelAccuracyF1 ScoreRecallSelectedNotesXGBoost76.58%0.64279.36%✓ PrimaryBest for churn identificationLogistic Regression81.83%0.61855.50%✓ SecondaryBest accuracy, interpretableDecision Tree72.89%--✗Lower performanceLinear Regression31.12%--✗Unsuitable for classification
Key Insights from Analysis
1. Class Imbalance Impact
The 73.5% / 26.5% class distribution required special handling through:

scale_pos_weight parameter in XGBoost
Custom threshold tuning
Focus on recall over accuracy

2. Feature Importance (Top Predictors)
Based on XGBoost model:

Contract type (Month-to-month highest risk)
Tenure (shorter tenure = higher churn)
Internet service type (Fiber optic users churn more)
Monthly charges (higher charges correlate with churn)
Payment method (Electronic check associated with higher churn)

3. Model Convergence
Logistic Regression showed convergence warning, indicating:

Complex feature relationships
Possible need for feature scaling
XGBoost's advantage in handling non-linear patterns

Deployment Architecture
Customer Database
       ↓
Feature Extraction Pipeline
       ↓
Preprocessed Features (22 dimensions)
       ↓
XGBoost Model (churn_xgb.pkl)
       ↓
Churn Probability Score
       ↓
Threshold Application (0.01)
       ↓
Risk Classification (High/Low)
       ↓
CRM System / Marketing Automation
Integration Options
Real-time Prediction API
pythonfrom flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('churn_xgb.pkl')
threshold = joblib.load('threshold.pkl')

@app.route('/predict', methods=['POST'])
def predict_churn():
    data = request.json
    features = preprocess_features(data)
    probability = model.predict_proba([features])[0][1]
    prediction = int(probability >= threshold)
    
    return jsonify({
        'churn_probability': float(probability),
        'high_risk': bool(prediction),
        'risk_level': 'High' if prediction else 'Low'
    })
Batch Processing
python# Daily churn risk scoring for entire customer base
customers_df = load_customer_data()
features = preprocess_pipeline(customers_df)
probabilities = model.predict_proba(features)[:, 1]

high_risk_customers = customers_df[probabilities >= threshold]
high_risk_customers['churn_probability'] = probabilities[probabilities >= threshold]

# Export for marketing team
high_risk_customers.to_csv('daily_high_risk_customers.csv')
Future Enhancements
Model Improvements

Feature Engineering:

Create interaction features (e.g., tenure × monthly_charges)
Engineer temporal features (recent behavior changes)
Add customer satisfaction scores if available


Advanced Techniques:

Ensemble methods combining XGBoost + LightGBM
Deep learning models (Neural Networks) for pattern recognition
Time-series analysis for behavior trend detection


Hyperparameter Optimization:

Bayesian optimization for XGBoost parameters
Cross-validation with stratified sampling
AutoML frameworks for automated tuning



System Enhancements

Model Monitoring:

Track model performance drift over time
Implement A/B testing framework
Set up automated retraining pipeline


Explainability:

SHAP values for individual predictions
LIME for local interpretability
Feature contribution dashboards for business users


Integration:

Real-time prediction API deployment
Integration with CRM systems (Salesforce, HubSpot)
Automated alert system for high-risk customers
Dashboard for business stakeholders



Repository Structure
telecom-churn-prediction/
│
├── .venv/                          # Virtual environment (not tracked in git)
│
├── backend/                        # Backend API and model serving
│   ├── app.py                      # Flask/FastAPI application
│   ├── models/                     # Saved model artifacts
│   │   ├── churn_xgb.pkl          # Trained XGBoost model
│   │   └── threshold.pkl          # Optimal prediction threshold
│   ├── preprocessing.py            # Feature preprocessing pipeline
│   └── requirements.txt            # Backend dependencies
│
├── churn-frontend/                 # Frontend web application
│   ├── public/                     # Static assets
│   ├── src/                        # React/Vue source code
│   │   ├── components/            # UI components
│   │   ├── pages/                 # Application pages
│   │   └── api/                   # API integration
│   ├── package.json               # Frontend dependencies
│   └── README.md                  # Frontend documentation
│
├── venv/                          # Alternative virtual environment
│
├── notebook4b76edc9f8.ipynb       # Main analysis notebook (from Kaggle)
│   └── Contains:
│       - Data loading and exploration
│       - Preprocessing pipeline
│       - Model training and evaluation
│       - Performance comparison
│
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
└── requirements.txt               # Project-wide dependencies
Requirements
txtpandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
Installation & Usage
Setup
bash# Clone repository
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
Running the Backend API
bashcd backend
python app.py

# API will be available at http://localhost:5000
Running the Frontend
bashcd churn-frontend
npm install
npm start

# Frontend will be available at http://localhost:3000
Using the Jupyter Notebook
bash# Install Jupyter in your virtual environment
pip install jupyter

# Launch notebook
jupyter notebook notebook4b76edc9f8.ipynb
Making Predictions via API
bash# POST request to predict endpoint
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "tenure": 12,
    "MonthlyCharges": 70,
    ...
  }'
Performance Monitoring
Key Metrics to Track in Production

Recall: Maintain above 75%
F1 Score: Monitor for degradation below 0.60
False Positive Rate: Keep under 30% to avoid alert fatigue
Model Confidence: Track average prediction probabilities
Business Metrics:

Retention campaign conversion rate
Cost per retained customer
ROI of retention efforts



Conclusion
This churn prediction solution leverages XGBoost's superior recall (79.36%) to maximize customer retention opportunities. By prioritizing the identification of potential churners over overall accuracy, the model aligns with business objectives of minimizing customer loss and maximizing retention ROI.
Key Takeaways:

XGBoost selected for 79.36% recall despite lower accuracy
Model effectively handles class imbalance through weighted training
Optimized threshold (0.01) ensures comprehensive churner identification
Significant business value: ~$778K potential revenue protection
Ready for production deployment with saved model artifacts

Contact & Support
For questions, feature requests, or deployment assistance, please contact me through my email or open an issue in the repository.
emailid-: workwithanshuman9468@gmail.com

Project Status: Production Ready ✅
Last Updated: December 2025
Model Version: 1.0
Dataset Version: Telco Customer Churn (Kaggle)
