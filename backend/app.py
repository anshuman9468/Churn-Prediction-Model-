from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all domains (Streamlit included)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("churn_xgb.pkl")
threshold = joblib.load("threshold.pkl")

# Input schema
class ChurnInput(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/predict")
def predict(data: ChurnInput):
    X = np.array(data.features).reshape(1, -1)
    proba = model.predict_proba(X)[:, 1][0]
    pred = int(proba >= threshold)
    return {
        "probability": float(proba),
        "prediction": pred
    }
