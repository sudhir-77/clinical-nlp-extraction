"""Peak RSS and a short hardware description."""

from __future__ import annotations

import os
import platform
import threading
import time

import psutil


class PeakMemory:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if total > self.peak_bytes:
            self.peak_bytes = total

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    @property
    def peak_gb(self) -> float:
        return round(self.peak_bytes / (1024 ** 3), 4)


def hardware_info(peak_gb: float, device: str) -> dict:
    memory = psutil.virtual_memory()
    cpu = platform.processor() or platform.machine()
    gpu = None
    if device.startswith("cuda"):
        gpu = device
    return {
        "cpu": cpu,
        "ram_gb": round(memory.total / (1024 ** 3), 2),
        "gpu": gpu,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": peak_gb,
    }
