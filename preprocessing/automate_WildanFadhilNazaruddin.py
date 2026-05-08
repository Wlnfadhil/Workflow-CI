from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "preprocessing"
    / "student_performance_processed_preprocessing.csv"
)

OUTPUT_DIR = BASE_DIR / "artifacts" / "datasets"

TARGET_COLUMN = "grade"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load processed dataset.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)

    print("\nDataset berhasil dimuat.")
    print(f"Shape dataset: {df.shape}")

    return df

# =========================================================
# SPLIT FEATURES & TARGET
# =========================================================

def split_features_target(df: pd.DataFrame):
    """
    Memisahkan fitur dan target.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

def split_train_test(X, y):
    """
    Membagi dataset menjadi train dan test.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nData berhasil dibagi.")
    print(f"Training data : {X_train.shape}")
    print(f"Testing data  : {X_test.shape}")

    return X_train, X_test, y_train, y_test

# =========================================================
# OVERSAMPLING
# =========================================================

def oversampling_data(X_train, y_train):
    """
    Melakukan oversampling hanya pada data training.
    """

    oversampler = RandomOverSampler(random_state=RANDOM_STATE)

    X_train_resampled, y_train_resampled = oversampler.fit_resample(
        X_train,
        y_train
    )

    print("\nOversampling berhasil dilakukan.")
    print(f"Shape X_train setelah oversampling: {X_train_resampled.shape}")

    return X_train_resampled, y_train_resampled

# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Menyimpan dataset hasil preprocessing.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("\nDataset berhasil disimpan.")
    print(f"Lokasi penyimpanan: {OUTPUT_DIR}")

# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # load dataset
    df = load_dataset(DATA_PATH)

    # split fitur dan target
    X, y = split_features_target(df)

    # train test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # oversampling training data
    X_train_resampled, y_train_resampled = oversampling_data(
        X_train,
        y_train
    )

    # save dataset
    save_dataset(
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test
    )

    print("\nPreprocessing automation completed successfully.")

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()