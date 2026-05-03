import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_ORDER = ['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5']

class Transaction(BaseModel):
    amount: float
    hour: int
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float

@app.get("/health")
def health():
    return {"status": "ok", "model": "fraud-detector v1"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.model_dump()])[FEATURE_ORDER]
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk": "high" if probability > 0.7 else "medium" if probability > 0.4 else "low",
        "action": "block" if probability > 0.7 else "review" if probability > 0.4 else "approve"
    }