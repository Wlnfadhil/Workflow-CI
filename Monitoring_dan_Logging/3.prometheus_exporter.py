"""Standalone Prometheus exporter for SMSML service-level metrics.

This exporter can be used when the inference API is served separately. It exposes
basic process metrics and synthetic service gauges on port 8001 by default.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
from prometheus_client import Counter, Gauge, start_http_server

EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8001"))
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "preprocessing/artifacts/models/best_xgboost.pkl"
    )
)

EXPORTER_UP = Gauge("smsml_exporter_up", "Exporter status. 1 means running.")
MODEL_FILE_EXISTS = Gauge("smsml_model_file_exists", "Model file availability. 1 means file exists.")
MODEL_FILE_SIZE_BYTES = Gauge("smsml_model_file_size_bytes", "Model file size in bytes.")
PROCESS_CPU_PERCENT = Gauge("smsml_process_cpu_percent", "Current process CPU usage percentage.")
PROCESS_MEMORY_BYTES = Gauge("smsml_process_memory_bytes", "Current process RSS memory usage in bytes.")
EXPORTER_HEARTBEATS = Counter("smsml_exporter_heartbeats_total", "Exporter heartbeat count.")


def collect_metrics() -> None:
    process = psutil.Process(os.getpid())
    EXPORTER_UP.set(1)
    PROCESS_CPU_PERCENT.set(process.cpu_percent(interval=None))
    PROCESS_MEMORY_BYTES.set(process.memory_info().rss)

    if MODEL_PATH.exists():
        MODEL_FILE_EXISTS.set(1)
        MODEL_FILE_SIZE_BYTES.set(MODEL_PATH.stat().st_size)
    else:
        MODEL_FILE_EXISTS.set(0)
        MODEL_FILE_SIZE_BYTES.set(0)

    EXPORTER_HEARTBEATS.inc()


def main() -> None:
    start_http_server(EXPORTER_PORT)
    print(f"Prometheus exporter berjalan di http://0.0.0.0:{EXPORTER_PORT}/metrics")

    while True:
        collect_metrics()
        time.sleep(5)


if __name__ == "__main__":
    main()