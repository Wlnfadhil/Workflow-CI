from pathlib import Path
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "preprocessing"
    / "student_performance_processed_preprocessing.csv"
)

ARTIFACT_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
)

DATASET_DIR = ARTIFACT_DIR / "datasets"
ENCODER_DIR = ARTIFACT_DIR / "encoders"
METADATA_DIR = ARTIFACT_DIR / "metadata"

TARGET_COLUMN = "grade"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================================================
# CREATE DIRECTORIES
# =========================================================

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ENCODER_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading raw dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset berhasil dimuat.")
print(df.head())

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

categorical_columns = [
    "gender",
    "internet_access",
    "extra_classes",
    "parent_education"
]

feature_encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col]
    )

    feature_encoders[col] = encoder

print("\nFeature categorical berhasil di-encode.")

# =========================================================
# ENCODE TARGET
# =========================================================

target_encoder = LabelEncoder()

df[TARGET_COLUMN] = target_encoder.fit_transform(
    df[TARGET_COLUMN]
)

print("\nTarget berhasil di-encode.")

# =========================================================
# SPLIT FEATURE & TARGET
# =========================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDataset berhasil dibagi.")

# =========================================================
# SCALING NUMERIC FEATURES
# =========================================================

numeric_columns = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_score",
    "midterm_score",
    "final_exam_score",
    "participation_score",
    "sleep_hours"
]

scaler = StandardScaler()

X_train[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

X_test[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

print("\nScaling berhasil dilakukan.")

# =========================================================
# SAVE DATASETS
# =========================================================

X_train.to_csv(
    DATASET_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    DATASET_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    DATASET_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    DATASET_DIR / "y_test.csv",
    index=False
)

print("\nDataset artifacts berhasil disimpan.")

# =========================================================
# SAVE ENCODERS
# =========================================================

joblib.dump(
    feature_encoders,
    ENCODER_DIR / "feature_label_encoders.pkl"
)

joblib.dump(
    target_encoder,
    ENCODER_DIR / "target_label_encoder.pkl"
)

joblib.dump(
    scaler,
    ENCODER_DIR / "scaler.joblib"
)

print("\nEncoder dan scaler berhasil disimpan.")

# =========================================================
# SAVE METADATA
# =========================================================

metadata = {

    "feature_columns":
        X.columns.tolist(),

    "categorical_columns":
        categorical_columns,

    "numeric_columns":
        numeric_columns,

    "target_classes":
        target_encoder.classes_.tolist()
}

with open(
    METADATA_DIR / "preprocessing_metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

print("\nMetadata berhasil disimpan.")

print("\nPreprocessing automation completed successfully.")