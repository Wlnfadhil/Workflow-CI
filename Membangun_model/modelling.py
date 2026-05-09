import argparse
import json
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
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

ARTIFACT_DIR = (
    BASE_DIR
    / "artifacts"
    / "mlflow_artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# DAGSHUB SWITCH
# =========================================================

USE_DAGSHUB = True

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
        "http://127.0.0.1:5000"
    )

    print("\nUsing Local MLflow Tracking")

print("\nTracking URI:")
print(mlflow.get_tracking_uri())

print("\nUSE_DAGSHUB:")
print(USE_DAGSHUB)

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
# + SAVE PREPROCESSING METADATA
# + SAVE ENCODERS
# =========================================================

import joblib

cat_cols = X_train.select_dtypes(
    include=["object"]
).columns

# =========================================================
# CREATE ARTIFACT DIRECTORIES
# =========================================================

METADATA_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
    / "metadata"
)

ENCODER_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
    / "encoders"
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ENCODER_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# FEATURE ENCODERS
# =========================================================

feature_encoders = {}

for col in cat_cols:

    encoder = LabelEncoder()

    X_train[col] = encoder.fit_transform(
        X_train[col]
    )

    X_test[col] = encoder.transform(
        X_test[col]
    )

    feature_encoders[col] = encoder

print("\nCategorical features berhasil di-encode.")

# =========================================================
# SAVE TARGET LABEL ENCODER
# =========================================================

target_encoder_path = (
    ENCODER_DIR
    / "target_label_encoder.pkl"
)

joblib.dump(
    label_encoder,
    target_encoder_path
)

print(
    "\nTarget label encoder berhasil disimpan."
)

# =========================================================
# SAVE FEATURE LABEL ENCODERS
# =========================================================

feature_encoder_path = (
    ENCODER_DIR
    / "feature_label_encoders.pkl"
)

joblib.dump(
    feature_encoders,
    feature_encoder_path
)

print(
    "\nFeature label encoders berhasil disimpan."
)

# =========================================================
# SAVE PREPROCESSING METADATA
# =========================================================

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

metadata = {

    "numeric_columns":
        numeric_columns,

    "categorical_columns":
        cat_cols.tolist(),

    "feature_columns":
        X_train.columns.tolist(),

    "target_classes":
        label_encoder.classes_.tolist()
}

metadata_path = (
    METADATA_DIR
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
# TRAINING
# =========================================================

with mlflow.start_run():

    # =====================================================
    # LOG PREPROCESSING ARTIFACTS TO MLFLOW
    # =====================================================

    mlflow.log_artifact(
        str(metadata_path)
    )

    mlflow.log_artifact(
        str(target_encoder_path)
    )

    mlflow.log_artifact(
        str(feature_encoder_path)
    )

    print(
        "\nMetadata dan encoder berhasil di-log ke MLflow."
    )

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

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
    # SAVE METRIC INFO JSON
    # =====================================================

    metric_info = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    metric_path = (
        ARTIFACT_DIR
        / "metric_info.json"
    )

    with open(metric_path, "w") as f:

        json.dump(
            metric_info,
            f,
            indent=4
        )

    mlflow.log_artifact(
        str(metric_path)
    )

    print("\nmetric_info.json berhasil disimpan.")

    # =====================================================
    # SAVE CONFUSION MATRIX IMAGE
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax)

    plt.title(
        "Training Confusion Matrix"
    )

    cm_path = (
        ARTIFACT_DIR
        / "training_confusion_matrix.png"
    )

    plt.savefig(cm_path)

    plt.close()

    mlflow.log_artifact(
        str(cm_path)
    )

    print(
        "\ntraining_confusion_matrix.png berhasil disimpan."
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

    print(
        "\nModel berhasil di-log ke MLflow."
    )

    print(
        "\nMLflow tracking completed successfully."
    )

# =========================================================
# RUN EXAMPLE
# =========================================================

# DagsHub mode
# python Membangun_model/modelling.py

# Localhost mode
# ubah:
 USE_DAGSHUB = False
#
# lalu jalankan:
 # python Membangun_model/modelling.py