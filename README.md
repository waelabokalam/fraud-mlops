# Fraud Detection MLOps Pipeline

End-to-end MLOps project detecting fraudulent transactions using a production-grade pipeline.

## The Problem
99.83% of transactions are normal. 0.17% are fraud. A dumb model that always says "not fraud" scores 99.83% accuracy and catches zero fraudsters. This project solves that.

## Solution
SMOTE (Synthetic Minority Oversampling) generates synthetic fraud examples to balance the dataset. The model learns real fraud patterns instead of just predicting "not fraud" every time.

## Stack
- **Model**: Random Forest (scikit-learn)
- **Balancing**: SMOTE (imbalanced-learn)
- **Experiment tracking**: MLflow
- **API**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 99.96% |
| Recall | 91.18% |
| Precision | 83.78% |
| F1 Score | 87.32% |

Catches 91% of all fraud cases.

## API Response
```json
{
  "fraud": true,
  "fraud_probability": 0.9998,
  "risk": "high",
  "action": "block"
}
```

Three actions: **block** (>70% fraud probability), **review** (40-70%), **approve** (<40%).

## Run locally

**Train the model:**
```bash
python3 src/train.py
```

**Start the API:**
```bash
uvicorn src.predict:app --port 8080
```

**Test a suspicious transaction:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 850.0, "hour": 2, "v1": -3.1, "v2": 3.2, "v3": -2.1, "v4": 2.3, "v5": -1.2}'
```

**Run with Docker:**
```bash
docker build -t fraud-mlops:v1 .
docker run -p 8080:8080 fraud-mlops:v1
```