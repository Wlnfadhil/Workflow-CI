import argparse
import json
from pathlib import Path

import dagshub
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
    roc_auc_score
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    BASE_DIR
    / "artifacts"
    / "datasets"
)

ARTIFACT_DIR = (
    BASE_DIR
    / "artifacts"
    / "tuning_artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# DAGSHUB SWITCH
# =========================================================

USE_DAGSHUB = False

# =========================================================
# ARGUMENT PARSER
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment_name",
    type=str,
    default="smsml-tuning"
)

args = parser.parse_args()

# =========================================================
# SETUP MLFLOW
# =========================================================

if USE_DAGSHUB:

    dagshub.init(
        repo_owner="Wlnfadhil",
        repo_name="SMSML_WildanFadhilNazaruddin_Dicoding",
        mlflow=True
    )

    mlflow.set_tracking_uri(
        "https://dagshub.com/Wlnfadhil/SMSML_WildanFadhilNazaruddin_Dicoding.mlflow"
    )

    print("\nUsing DagsHub MLflow Tracking")

else:

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    print("\nUsing Local File MLflow Tracking")

mlflow.set_experiment(
    args.experiment_name
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading tuning artifacts...")

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
# SAVE PREPROCESSING METADATA
# =========================================================

metadata_dir = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
    / "metadata"
)

metadata_dir.mkdir(
    parents=True,
    exist_ok=True
)

metadata = {

    "feature_columns":
        X_train.columns.tolist(),

    "categorical_columns":
        cat_cols.tolist()
}

metadata_path = (
    metadata_dir
    / "preprocessing_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

print(
    "\npreprocessing_metadata.json berhasil disimpan."
)
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

                print("\n===================================")
                print(f"Training : {run_name}")
                print("===================================")

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
                # LOG PARAMETERS
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
                # LOG METRICS
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
                # SAVE BEST CONFIGURATION
                # =====================================

                if accuracy > best_accuracy:

                    best_accuracy = accuracy

                    best_configuration = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "roc_auc": roc_auc
                    }

# =========================================================
# SAVE BEST CONFIGURATION
# =========================================================

best_config_path = (
    ARTIFACT_DIR
    / "best_tuning_configuration.json"
)

with open(best_config_path, "w") as f:

    json.dump(
        best_configuration,
        f,
        indent=4
    )

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

print("\n===================================")
print("BEST TUNING CONFIGURATION")
print("===================================")

print(best_configuration)

print(
    "\nBest tuning configuration "
    "berhasil disimpan."
)

print(
    "\nMLflow tuning completed successfully."
)

# =========================================================
# RUN EXAMPLE
# =========================================================

# DagsHub mode
# python Membangun_model/modelling_tuning.py

# Localhost mode
# ubah:
 # USE_DAGSHUB = False
#
# lalu jalankan:
# python Membangun_model/modelling_tuning.py