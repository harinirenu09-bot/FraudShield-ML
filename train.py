
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# 1. Project folders
DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 2. Load a manageable sample for our first training run
print("Loading transaction data...")
df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFraud"
    ],
    nrows=200_000,
)

print("Rows loaded:", len(df))
print("Fraud counts:")
print(df["isFraud"].value_counts())

# 3. Separate features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

numeric_features = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("transaction_type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ("numeric", "passthrough", numeric_features),
    ]
)

# 4. Split data while preserving the fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Create the XGBoost model
fraud_count = int(y_train.sum())
normal_count = len(y_train) - fraud_count

if fraud_count == 0:
    raise ValueError("No fraud cases found. Try using more dataset rows.")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=normal_count / fraud_count,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )),
])

# 6. Train and track the experiment in MLflow
mlflow.set_experiment("FraudShield-ML")

with mlflow.start_run():
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    avg_precision = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("rows_used", len(df))
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("average_precision", avg_precision)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
    model,
    name="fraud_detection_model",
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
    ],
)
    # 7. Save the trained model for our web app
    joblib.dump(model, MODEL_DIR / "fraud_model.joblib")

    print("\nModel training completed!")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nModel saved to models/fraud_model.joblib")
    print("MLflow experiment: FraudShield-ML")
