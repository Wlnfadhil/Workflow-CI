import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from mlflow.models.signature import infer_signature

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    BASE_DIR
    / "artifacts"
    / "datasets"
)

# =========================================================
# ARGUMENT PARSER
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_estimators",
    type=int,
    default=200
)

parser.add_argument(
    "--max_depth",
    type=int,
    default=10
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.1
)

args = parser.parse_args()

# =========================================================
# SETUP MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "smsml-experiment"
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading training artifacts...")

X_train = pd.read_csv(
    TRAIN_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    TRAIN_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    TRAIN_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    TRAIN_DIR / "y_test.csv"
).squeeze()

print("\nDataset berhasil dimuat.")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# =========================================================
# ENCODE TARGET
# =========================================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    y_train
)

y_test = label_encoder.transform(
    y_test
)

print("\nTarget berhasil di-encode.")

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

cat_cols = X_train.select_dtypes(
    include=["object"]
).columns

for col in cat_cols:

    encoder = LabelEncoder()

    X_train[col] = encoder.fit_transform(
        X_train[col]
    )

    X_test[col] = encoder.transform(
        X_test[col]
    )

print("\nCategorical features berhasil di-encode.")

# =========================================================
# TRAINING
# =========================================================

with mlflow.start_run():

    print("\nTraining model XGBoost...")

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
        eval_metric="mlogloss"
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
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

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr"
    )

    # =====================================================
    # PRINT RESULT
    # =====================================================

    print("\n===== EVALUATION RESULT =====")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\n===== CONFUSION MATRIX =====")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # =====================================================
    # LOG PARAMETERS
    # =====================================================

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "n_estimators",
        args.n_estimators
    )

    mlflow.log_param(
        "max_depth",
        args.max_depth
    )

    mlflow.log_param(
        "learning_rate",
        args.learning_rate
    )

    # =====================================================
    # LOG METRICS
    # =====================================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    # =====================================================
    # MODEL SIGNATURE
    # =====================================================

    signature = infer_signature(
        X_train,
        y_pred
    )

    # =====================================================
    # LOG MODEL TO MLFLOW
    # =====================================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature
    )

    print("\nModel berhasil di-log ke MLflow.")

    print(
        "\nMLflow tracking completed successfully."
    )

# =========================================================
# RUN EXAMPLE
# =========================================================

# python Membangun_model/modelling.py
#
# python Membangun_model/modelling.py \
# --n_estimators 300 \
# --max_depth 15 \
# --learning_rate 0.05