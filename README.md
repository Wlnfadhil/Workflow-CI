# SMSML — Student Performance Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

# Project Overview

SMSML (Student Monitoring System Machine Learning) adalah proyek end-to-end MLOps pipeline untuk prediksi performa akademik siswa menggunakan machine learning berbasis XGBoost.

Project ini mencakup:

- Automated preprocessing
- MLflow experiment tracking
- Hyperparameter tuning
- FastAPI inference serving
- Prometheus monitoring
- Grafana visualization dashboard
- CI/CD GitHub Actions
- Docker-ready deployment architecture

---

# Machine Learning Objective

Model digunakan untuk memprediksi performa akademik siswa berdasarkan:

- jam belajar,
- kehadiran,
- nilai tugas,
- nilai ujian,
- partisipasi,
- akses internet,
- pendidikan orang tua,
- dan faktor pendukung lainnya.

---

# Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn, XGBoost |
| Experiment Tracking | MLflow |
| API Serving | FastAPI |
| Monitoring | Prometheus |
| Visualization | Grafana |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Dataset Processing | Pandas, NumPy |
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
│   └── tuning_artifacts/
│
├── mlruns/
│
├── .github/
│   └── workflows/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE