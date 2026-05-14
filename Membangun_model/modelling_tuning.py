from pathlib import Path
import json

import dagshub
import mlflow
import mlflow.sklearn

import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

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

ARTIFACT_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
    / "tuning_artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPERIMENT_NAME = "smsml-xgboost-tuning"

# =========================================================
# DAGSHUB MLFLOW SETUP
# =========================================================

dagshub.init(
    repo_owner="Wlnfadhil",
    repo_name="SMSML_WildanFadhilNazaruddin_Dicoding",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Wlnfadhil/SMSML_WildanFadhilNazaruddin_Dicoding.mlflow"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

print("\nUsing DagsHub MLflow Tracking")

# =========================================================
# LOAD DATASET ARTIFACTS
# =========================================================

print("\nLoading preprocessing artifacts...")

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
# HYPERPARAMETER SEARCH SPACE
# =========================================================

n_estimators_list = [
    100,
    200,
    300
]

max_depth_list = [
    5,
    10,
    15
]

learning_rate_list = [
    0.01,
    0.05,
    0.1
]

# =========================================================
# BEST MODEL TRACKING
# =========================================================

best_accuracy = 0

best_configuration = {}

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

for n_estimators in n_estimators_list:

    for max_depth in max_depth_list:

        for learning_rate in learning_rate_list:

            run_name = (
                f"xgb_"
                f"n{n_estimators}_"
                f"d{max_depth}_"
                f"lr{learning_rate}"
            )

            with mlflow.start_run(
                run_name=run_name
            ):

                print("\n" + "=" * 50)
                print(f"Training : {run_name}")
                print("=" * 50)

                # =====================================
                # MODEL
                # =====================================

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    eval_metric="mlogloss"
                )

                # =====================================
                # TRAINING
                # =====================================

                model.fit(
                    X_train,
                    y_train
                )

                # =====================================
                # PREDICTION
                # =====================================

                y_pred = model.predict(
                    X_test
                )

                y_prob = model.predict_proba(
                    X_test
                )

                # =====================================
                # EVALUATION
                # =====================================

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

                # =====================================
                # PRINT RESULT
                # =====================================

                print(f"Accuracy  : {accuracy:.4f}")
                print(f"Precision : {precision:.4f}")
                print(f"Recall    : {recall:.4f}")
                print(f"F1 Score  : {f1:.4f}")
                print(f"ROC AUC   : {roc_auc:.4f}")

                # =====================================
                # MANUAL LOGGING PARAMETERS
                # =====================================

                mlflow.log_param(
                    "model",
                    "XGBoost"
                )

                mlflow.log_param(
                    "n_estimators",
                    n_estimators
                )

                mlflow.log_param(
                    "max_depth",
                    max_depth
                )

                mlflow.log_param(
                    "learning_rate",
                    learning_rate
                )

                # =====================================
                # MANUAL LOGGING METRICS
                # =====================================

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

                # =====================================
                # LOG MODEL
                # =====================================

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )

                # =====================================
                # SAVE BEST CONFIGURATION
                # =====================================

                if accuracy > best_accuracy:

                    best_accuracy = accuracy

                    best_configuration = {

                        "n_estimators":
                            n_estimators,

                        "max_depth":
                            max_depth,

                        "learning_rate":
                            learning_rate,

                        "accuracy":
                            accuracy,

                        "precision":
                            precision,

                        "recall":
                            recall,

                        "f1_score":
                            f1,

                        "roc_auc":
                            roc_auc
                    }

# =========================================================
# SAVE BEST CONFIGURATION JSON
# =========================================================

best_config_path = (
    ARTIFACT_DIR
    / "best_tuning_configuration.json"
)

with open(
    best_config_path,
    "w"
) as file:

    json.dump(
        best_configuration,
        file,
        indent=4
    )

print("\nBest tuning configuration disimpan.")

# =========================================================
# LOG ADDITIONAL ARTIFACT
# =========================================================

with mlflow.start_run(
    run_name="best_tuning_summary"
):

    mlflow.log_artifact(
        str(best_config_path)
    )

# =========================================================
# FINAL RESULT
# =========================================================

print("\n" + "=" * 50)
print("BEST TUNING CONFIGURATION")
print("=" * 50)

print(best_configuration)

print("\nMLflow tuning completed successfully.")