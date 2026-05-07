import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# ========================
# 1. Argument Parser
# ========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="data/student_performance_data.csv"
)

args = parser.parse_args()

# ========================
# 2. Setup MLflow
# ========================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("smsml-tuning")

# ========================
# 3. Load Dataset
# ========================
df = pd.read_csv(args.dataset)

# Drop leakage column
df = df.drop(columns=["overall_score", "student_id"], errors="ignore")

# ========================
# 4. Split Feature & Target
# ========================
y = df["grade"]
X = df.drop("grade", axis=1)

# ========================
# 5. Encoding
# ========================
X = pd.get_dummies(X)

# ========================
# 6. Train Test Split
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ========================
# 7. Hyperparameter Range
# ========================
n_estimators_range = [50, 100, 200]
max_depth_range = [5, 10, 20]

# ========================
# 8. Best Model Tracking
# ========================
best_accuracy = 0
best_params = {}
best_model = None

# ========================
# 9. Hyperparameter Tuning
# ========================
for n in n_estimators_range:
    for d in max_depth_range:

        with mlflow.start_run(run_name=f"rf_{n}_{d}"):

            # Model
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                random_state=42
            )

            # Training
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # ========================
            # Evaluation Metrics
            # ========================
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
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
            # Log Params
            # ========================
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)

            # ========================
            # Log Metrics
            # ========================
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # ========================
            # Log Model
            # ========================
            mlflow.sklearn.log_model(model, "model")

            print(f"n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

            # ========================
            # Save Best Model
            # ========================
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    "n_estimators": n,
                    "max_depth": d
                }
                best_model = model

# ========================
# 10. Final Best Model Info
# ========================
print("\n===== BEST MODEL =====")
print(f"Best Accuracy : {best_accuracy:.4f}")
print(f"Best Params   : {best_params}")

# ========================
# 11. Save Best Model
# ========================
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.joblib"
)

print("\nBest model berhasil disimpan!")
print("Lokasi: models/best_model.joblib")