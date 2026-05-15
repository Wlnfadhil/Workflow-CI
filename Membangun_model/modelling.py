import os
from pathlib import Path

import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
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

mlflow.set_experiment(
    "smsml-basic-experiment"
)
# =========================================================
# AUTOLOG
# =========================================================

mlflow.sklearn.autolog(
    log_models=True
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

    with mlflow.start_run(
        run_name=model_name
    ):

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
        # EXPLICIT MODEL LOGGING
        # =================================================

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(
            f"{model_name} artifact berhasil disimpan."
        )

print("\nMLflow modelling completed successfully.")