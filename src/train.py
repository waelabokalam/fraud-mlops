import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, classification_report
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/fraud.csv")
print(f"Data loaded: {df.shape[0]} rows")
print(f"Fraud cases: {df['class'].sum()} ({df['class'].mean()*100:.2f}%)")

# ── 2. SPLIT FEATURES AND TARGET ─────────────────────────────────────────────
X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain fraud cases: {y_train.sum()}")
print(f"Test fraud cases: {y_test.sum()}")

# ── 3. APPLY SMOTE ───────────────────────────────────────────────────────────
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {y_train.sum()} fraud cases")
print(f"After SMOTE: {y_train_balanced.sum()} fraud cases")

# ── 4. TRAIN WITH MLFLOW ─────────────────────────────────────────────────────
mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="baseline_smote"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("smote", True)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(model, "model")

    print(f"\n── Results ──────────────────────")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])}")

    # ── 5. SAVE MODEL ─────────────────────────────────────────────────────────────
import pickle
import os
os.makedirs("models", exist_ok=True)
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to models/fraud_model.pkl")
print(f"Feature order: {list(X.columns)}")