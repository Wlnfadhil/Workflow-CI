import os
from pathlib import Path

import joblib
import pandas as pd

import mlflow

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "datasets"
)

# =========================================================
# MLFLOW SETUP
# =========================================================

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(
    TRACKING_URI
)

# =========================================================
# DISABLE AUTOLOG
# =========================================================

mlflow.autolog(
    disable=True
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset artifacts...")

X_train = pd.read_csv(
    DATASET_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    DATASET_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATASET_DIR / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    DATASET_DIR / "y_test.csv"
).squeeze()

print("\nDataset berhasil dimuat.")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# =========================================================
# MODELS
# =========================================================

models = {

    "LogisticRegression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "DecisionTree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            random_state=42
        ),

    "GaussianNB":
        GaussianNB(),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="mlogloss"
        )
}

# =========================================================
# TRAINING LOOP
# =========================================================

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL : {model_name}")
    print("=" * 60)

    # =====================================================
    # START NESTED RUN
    # =====================================================

    with mlflow.start_run(
        run_name=model_name,
        nested=True
    ):

        # =================================================
        # TAG
        # =================================================

        mlflow.set_tag(
            "model_name",
            model_name
        )

        # =================================================
        # TRAINING
        # =================================================

        model.fit(
            X_train,
            y_train
        )

        # =================================================
        # PREDICTION
        # =================================================

        y_pred = model.predict(
            X_test
        )

        # =================================================
        # EVALUATION
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        # =================================================
        # LOG METRICS
        # =================================================

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        confusion_matrix_path = (
            f"{model_name}_confusion_matrix.png"
        )

        plt.savefig(
            confusion_matrix_path,
            bbox_inches="tight"
        )

        mlflow.log_artifact(
            confusion_matrix_path,
            artifact_path="evaluation"
        )

        plt.close()

        # =================================================
        # SAVE MODEL MANUALLY
        # =================================================

        model_path = (
            f"{model_name}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        # =================================================
        # LOG MODEL ARTIFACT
        # =================================================

        mlflow.log_artifact(
            model_path,
            artifact_path="model"
        )

        print(
            f"{model_name} artifact berhasil disimpan."
        )

print("\nMLflow modelling completed successfully.")