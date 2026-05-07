# smsml-project

Project scaffold for a reproducible MLflow-based machine learning pipeline.

## Current setup
- OS: WSL Ubuntu
- Python: Conda env `mlops`
- Tracking: MLflow at `http://127.0.0.1:5000`
- Working directory: `/home/wldn/ml-projects/smsml-project`

## Next steps
1. Extract `data/student-performance-analytics-dataset.zip` into `data/raw/`.
2. Inspect the dataset columns and define the target variable.
3. Implement a preprocessing pipeline under `src/`.
4. Add training with MLflow logging, parameters, metrics, and artifacts.
5. Add hyperparameter tuning with Optuna or scikit-learn search.
6. Save the best model and document the reproducible run command.

## Dependency install
```bash
conda activate mlops
pip install -r requirements.txt
```
