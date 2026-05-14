"""Standalone Prometheus exporter for SMSML monitoring."""

from __future__ import annotations

import os
import time

from pathlib import Path

import psutil

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
)

EXPORTER_PORT = int(
    os.getenv(
        "EXPORTER_PORT",
        "8001"
    )
)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            ARTIFACT_DIR
            / "models"
            / "best_model.joblib"
        )
    )
)

METADATA_PATH = Path(
    os.getenv(
        "METADATA_PATH",
        str(
            ARTIFACT_DIR
            / "metadata"
            / "preprocessing_metadata.json"
        )
    )
)

# =========================================================
# PROMETHEUS METRICS
# =========================================================

EXPORTER_UP = Gauge(
    "smsml_exporter_up",
    "Exporter running status."
)

MODEL_FILE_EXISTS = Gauge(
    "smsml_model_file_exists",
    "Model file existence status."
)

MODEL_FILE_SIZE_BYTES = Gauge(
    "smsml_model_file_size_bytes",
    "Model file size in bytes."
)

METADATA_FILE_EXISTS = Gauge(
    "smsml_metadata_file_exists",
    "Metadata file existence status."
)

PROCESS_CPU_PERCENT = Gauge(
    "smsml_process_cpu_percent",
    "Current process CPU usage percentage."
)

PROCESS_MEMORY_BYTES = Gauge(
    "smsml_process_memory_bytes",
    "Current process memory usage in bytes."
)

PROCESS_THREADS = Gauge(
    "smsml_process_threads_total",
    "Total running process threads."
)

EXPORTER_HEARTBEATS = Counter(
    "smsml_exporter_heartbeats_total",
    "Exporter heartbeat counter."
)

EXPORTER_COLLECTION_LATENCY = Histogram(
    "smsml_exporter_collection_latency_seconds",
    "Metric collection latency."
)

# =========================================================
# METRIC COLLECTION
# =========================================================

def collect_metrics() -> None:

    start_time = time.perf_counter()

    process = psutil.Process(
        os.getpid()
    )

    # =============================================
    # EXPORTER STATUS
    # =============================================

    EXPORTER_UP.set(1)

    # =============================================
    # PROCESS METRICS
    # =============================================

    PROCESS_CPU_PERCENT.set(
        process.cpu_percent(interval=None)
    )

    PROCESS_MEMORY_BYTES.set(
        process.memory_info().rss
    )

    PROCESS_THREADS.set(
        process.num_threads()
    )

    # =============================================
    # MODEL FILE METRICS
    # =============================================

    if MODEL_PATH.exists():

        MODEL_FILE_EXISTS.set(1)

        MODEL_FILE_SIZE_BYTES.set(
            MODEL_PATH.stat().st_size
        )

    else:

        MODEL_FILE_EXISTS.set(0)

        MODEL_FILE_SIZE_BYTES.set(0)

    # =============================================
    # METADATA FILE METRICS
    # =============================================

    if METADATA_PATH.exists():

        METADATA_FILE_EXISTS.set(1)

    else:

        METADATA_FILE_EXISTS.set(0)

    # =============================================
    # HEARTBEAT
    # =============================================

    EXPORTER_HEARTBEATS.inc()

    # =============================================
    # COLLECTION LATENCY
    # =============================================

    latency = (
        time.perf_counter()
        - start_time
    )

    EXPORTER_COLLECTION_LATENCY.observe(
        latency
    )

# =========================================================
# MAIN
# =========================================================

def main() -> None:

    start_http_server(
        EXPORTER_PORT
    )

    print("=" * 60)
    print("SMSML PROMETHEUS EXPORTER STARTED")
    print("=" * 60)

    print(
        f"Metrics endpoint : "
        f"http://0.0.0.0:{EXPORTER_PORT}/metrics"
    )

    print(
        f"Model path : {MODEL_PATH}"
    )

    print(
        f"Metadata path : {METADATA_PATH}"
    )

    print("=" * 60)

    while True:

        collect_metrics()

        time.sleep(5)

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()