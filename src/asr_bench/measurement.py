"""Wall-clock phase timing and sampled process memory (not system-wide memory)."""

import threading
import time
from contextlib import contextmanager

import psutil


class Measurement:
    def __init__(self, device: str):
        self.device = device
        self.timings = {}
        self.peak_rss = 0
        self._stop = threading.Event()

    def __enter__(self):
        if self.device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)

    @contextmanager
    def phase(self, name):
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[name] = time.perf_counter() - started

    def report(self):
        gpu = None
        if self.device == "cuda":
            import torch

            gpu = torch.cuda.max_memory_allocated() / 2**20
        return {
            "seconds": self.timings,
            "peak_process_rss_mib": self.peak_rss / 2**20,
            "peak_cuda_allocated_mib": gpu,
            "rss_sampling_interval_ms": 20,
        }

    def _sample(self):
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self._stop.wait(0.02)

    def _sync(self):
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()
