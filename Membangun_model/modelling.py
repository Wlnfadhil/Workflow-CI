import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder

import argparse

# ========================
# 1. Argument Parser
# ========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="data/student_performance_data.csv"
)

parser.add_argument(
    "--n_estimators",
    type=int,
    default=100
)

parser.add_argument(
    "--max_depth",
    type=int,
    default=10
)

args = parser.parse_args()

# ========================
# 2. Setup MLflow
# ========================
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("smsml-experiment")

# ========================
# 3. Load data
# ========================
df = pd.read_csv(args.dataset)

# drop leakage
df = df.drop(
    columns=["overall_score", "student_id"],
    errors="ignore"
)

# ========================
# 4. Encoding
# ========================
cat_cols = df.select_dtypes(
    include=["object", "category"]
).columns

le_dict = {}

for col in cat_cols:
    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    le_dict[col] = le

# ========================
# 5. Split data
# ========================
X = df.drop("grade", axis=1)
y = df["grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ========================
# 6. Training + Logging
# ========================
with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # ========================
    # Metrics
    # ========================
    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    # ========================
    # Log Parameters
    # ========================
    mlflow.log_param("model", "RandomForest")

    mlflow.log_param(
        "dataset",
        args.dataset
    )

    mlflow.log_param(
        "n_estimators",
        args.n_estimators
    )

    mlflow.log_param(
        "max_depth",
        args.max_depth
    )

    # ========================
    # Log Metrics
    # ========================
    mlflow.log_metric("accuracy", acc)

    mlflow.log_metric(
        "f1_macro",
        f1
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    # ========================
    # Log Model
    # ========================
    mlflow.sklearn.log_model(
        model,
        "model"
    )

    # ========================
    # Output
    # ========================
    print("\n===== TRAINING SELESAI =====")

    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 Macro      : {f1:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")

    print("\n===== PARAMETER =====")

    print(f"Dataset       : {args.dataset}")
    print(f"n_estimators  : {args.n_estimators}")
    print(f"max_depth     : {args.max_depth}")

# ========================
# Run Example
# ========================
# python Membangun_model/modelling.py
#
# python Membangun_model/modelling.py \
# --dataset data/student_performance_data.csv \
# --n_estimators 200 \
# --max_depth 20