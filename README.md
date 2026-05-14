# SMSML — Student Performance Prediction MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)
![ML](https://img.shields.io/badge/Machine_Learning-XGBoost-success)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

</div>

---

# Project Overview

SMSML (Student Monitoring System Machine Learning) merupakan proyek end-to-end MLOps pipeline untuk memprediksi performa akademik siswa menggunakan machine learning berbasis XGBoost.

Project ini tidak hanya berfokus pada model machine learning, tetapi juga implementasi lifecycle MLOps secara menyeluruh:

- automated preprocessing,
- experiment tracking,
- hyperparameter tuning,
- API inference serving,
- observability,
- monitoring,
- alerting,
- CI/CD automation,
- hingga deployment-ready architecture.

---

# Machine Learning Objective

Model digunakan untuk memprediksi performa akademik siswa berdasarkan berbagai faktor akademik dan non-akademik seperti:

- study hours,
- attendance percentage,
- assignment score,
- midterm score,
- final exam score,
- participation score,
- internet access,
- extra classes,
- parental education,
- sleep hours,
- dan faktor pendukung lainnya.

---

# MLOps Architecture

```text
Dataset
   │
   ▼
Automated Preprocessing
   │
   ▼
Model Training (MLflow)
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Model Artifact Logging
   │
   ▼
FastAPI Inference API
   │
   ▼
Prometheus Monitoring
   │
   ▼
Grafana Dashboard & Alerting
   │
   ▼
CI/CD GitHub Actions
```

---

# Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.12 |
| Machine Learning | Scikit-Learn |
| Boosting Algorithm | XGBoost |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Monitoring | Prometheus |
| Visualization | Grafana |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Dataset Processing | Pandas & NumPy |
| Oversampling | Imbalanced-Learn |

---

# Project Structure

```text
SMSML_WildanFadhilNazaruddin_Dicoding/
│
├── preprocessing/
│   ├── automate_WildanFadhilNazaruddin.py
│   └── artifacts/
│
├── Membangun_model/
│   ├── modelling.py
│   └── modelling_tuning.py
│
├── Monitoring_dan_Logging/
│   ├── 07.Inference.py
│   ├── 2.prometheus.yml
│   └── 3.prometheus_exporter.py
│
├── artifacts/
│   ├── datasets/
│   ├── mlflow_artifacts/
│   └── tuning_artifacts/
│
├── mlruns/
│
├── .github/
│   └── workflows/
│       └── blank.yml
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

---

# Automated Preprocessing

Pipeline preprocessing meliputi:

- train-test split,
- oversampling untuk imbalance class,
- categorical encoding,
- metadata generation,
- feature encoder persistence,
- artifact generation.

Menjalankan preprocessing:

```bash
python preprocessing/automate_WildanFadhilNazaruddin.py
```

---

# Model Training

Training model menggunakan:

- XGBoost Classifier,
- MLflow experiment tracking,
- automatic metric logging,
- artifact persistence,
- confusion matrix logging,
- model signature tracking.

Menjalankan baseline training:

```bash
python Membangun_model/modelling.py
```

---

# Hyperparameter Tuning

Pipeline tuning menggunakan kombinasi hyperparameter:

- n_estimators,
- max_depth,
- learning_rate.

Semua hasil tuning dicatat otomatis menggunakan MLflow.

Menjalankan hyperparameter tuning:

```bash
python Membangun_model/modelling_tuning.py
```

---

# MLflow Experiment Tracking

Project menggunakan MLflow untuk:

- experiment tracking,
- model versioning,
- metric logging,
- artifact management,
- parameter comparison.

Menjalankan MLflow UI:

```bash
mlflow ui
```

Default access:

```text
http://127.0.0.1:5000
```

---

# FastAPI Inference Serving

Inference API dibangun menggunakan FastAPI.

Menjalankan inference API:

```bash
python Monitoring_dan_Logging/07.Inference.py
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Metrics endpoint:

```text
http://127.0.0.1:8000/metrics
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

---

# Example Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "study_hours_per_day": 6.5,
  "attendance_percentage": 88,
  "assignment_score": 80,
  "midterm_score": 78,
  "final_exam_score": 82,
  "participation_score": 75,
  "sleep_hours": 7,
  "gender": "Male",
  "internet_access": "Yes",
  "extra_classes": "No",
  "parent_education": "Bachelor"
}'
```

---

# Monitoring with Prometheus

Prometheus digunakan untuk monitoring:

- total inference requests,
- request latency,
- CPU usage,
- memory usage,
- model metrics,
- exporter heartbeat.

Menjalankan Prometheus:

```bash
prometheus \
--config.file=Monitoring_dan_Logging/2.prometheus.yml \
--storage.tsdb.path=./prometheus_data
```

Default access:

```text
http://localhost:9090
```

---

# Visualization with Grafana

Grafana digunakan untuk visualisasi dashboard monitoring:

- CPU dashboard,
- memory dashboard,
- request dashboard,
- inference latency,
- alerting dashboard,
- Prometheus visualization.

Menjalankan Grafana:

```bash
sudo systemctl start grafana-server
```

Default access:

```text
http://localhost:3000
```

---

# Implemented Monitoring Metrics

Monitoring metrics yang telah diimplementasikan:

- `smsml_inference_requests_total`
- `smsml_inference_latency_seconds`
- `smsml_prediction_count_total`
- `smsml_model_loaded`
- `smsml_process_cpu_percent`
- `smsml_process_memory_bytes`
- `smsml_exporter_heartbeats_total`

---

# Alerting System

Project juga mengimplementasikan alerting menggunakan Grafana Alert Rules.

Contoh alert:

- High CPU Usage
- High Memory Usage
- API Down Detection
- High Request Latency

---

# CI/CD Pipeline

GitHub Actions digunakan untuk automation pipeline:

- dependency installation,
- automated preprocessing,
- model training,
- hyperparameter tuning,
- artifact upload,
- Docker build preparation.

Workflow file:

```text
.github/workflows/blank.yml
```

---

# Docker Support

Project sudah disiapkan untuk containerized deployment menggunakan Docker.

Build Docker image:

```bash
docker build -t smsml-model .
```

Run Docker container:

```bash
docker run -p 8000:8000 smsml-model
```

---

# Screenshots & Submission Evidence

Repository menyertakan bukti implementasi:

- inference API,
- Swagger UI,
- prediction endpoint,
- Prometheus monitoring,
- Grafana dashboard,
- alerting configuration,
- CI/CD GitHub Actions,
- monitoring metrics,
- model serving evidence.

---

# Key MLOps Features

✅ Automated preprocessing  
✅ Experiment tracking  
✅ Hyperparameter tuning  
✅ API serving  
✅ Monitoring system  
✅ Alerting system  
✅ CI/CD automation  
✅ Docker-ready deployment  
✅ Artifact management  
✅ Production-ready structure  

---

# Future Improvements

Beberapa pengembangan yang dapat dilakukan selanjutnya:

- Kubernetes deployment,
- model registry,
- automated retraining,
- distributed inference,
- drift detection,
- real-time streaming inference,
- centralized logging.

---

# Author

**Wildan Fadhil Nazaruddin**

Machine Learning & MLOps Enthusiast

---

# License

This project is licensed under the MIT License.