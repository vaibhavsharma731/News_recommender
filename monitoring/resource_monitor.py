"""
Resource snapshot and background monitoring for NewsRec v2.0.

Design principles:
  - psutil snapshots are taken AFTER the work, never polled during live recommendations.
  - GPU metrics use torch.cuda / pynvml when available; clearly None when unavailable.
  - BackgroundResourceMonitor runs a daemon thread at 500ms intervals for long jobs only.
  - Every unavailable metric is None — never 0 — so callers can distinguish "unavailable"
    from genuine zero usage.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil

# Optional GPU support
try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ─── Snapshot dataclass ───────────────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    """Point-in-time resource reading. Fields are None when genuinely unavailable."""
    timestamp: float = field(default_factory=time.time)

    # CPU
    cpu_percent: Optional[float] = None        # process CPU % (cumulative, not instantaneous)
    cpu_time_user: Optional[float] = None      # seconds of user CPU time
    cpu_time_sys: Optional[float] = None       # seconds of system CPU time

    # RAM
    ram_mb: Optional[float] = None             # current RSS in MB
    ram_vms_mb: Optional[float] = None         # virtual memory size in MB
    ram_peak_mb: Optional[float] = None        # peak RSS observed (None unless tracked externally)

    # GPU (all None on a CPU-only machine)
    gpu_available: bool = False
    gpu_util_pct: Optional[float] = None       # 0–100 GPU utilisation
    gpu_mem_used_mb: Optional[float] = None    # allocated GPU memory in MB
    gpu_mem_total_mb: Optional[float] = None   # total GPU memory in MB
    gpu_mem_peak_mb: Optional[float] = None    # peak allocated (tracked externally)

    # Disk I/O since process start (cumulative)
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
        )
        return d


# ─── Single snapshot ──────────────────────────────────────────────────────────

def take_snapshot() -> ResourceSnapshot:
    """
    Take a lightweight point-in-time resource snapshot of the current process.
    Safe to call from any thread. Never raises.
    """
    snap = ResourceSnapshot()
    try:
        proc = psutil.Process(os.getpid())
        with proc.oneshot():
            try:
                snap.cpu_percent = proc.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                ct = proc.cpu_times()
                snap.cpu_time_user = round(ct.user, 3)
                snap.cpu_time_sys = round(ct.system, 3)
            except Exception:
                pass
            try:
                mem = proc.memory_info()
                snap.ram_mb = round(mem.rss / 1024 / 1024, 2)
                snap.ram_vms_mb = round(mem.vms / 1024 / 1024, 2)
            except Exception:
                pass
            try:
                io = proc.io_counters()
                snap.disk_read_mb = round(io.read_bytes / 1024 / 1024, 3)
                snap.disk_write_mb = round(io.write_bytes / 1024 / 1024, 3)
            except Exception:
                pass  # unavailable on some platforms/OSes

    except Exception:
        pass  # process lookup failure — return what we have

    # GPU via torch.cuda
    if _TORCH_OK and torch.cuda.is_available():
        snap.gpu_available = True
        try:
            snap.gpu_mem_used_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
            snap.gpu_mem_total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2)
        except Exception:
            pass

    # GPU utilisation via pynvml (needs root/driver access — optional)
    if _NVML_OK:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snap.gpu_util_pct = float(util.gpu)
            if snap.gpu_mem_total_mb is None:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                snap.gpu_mem_used_mb = round(mem_info.used / 1024 / 1024, 2)
                snap.gpu_mem_total_mb = round(mem_info.total / 1024 / 1024, 2)
            snap.gpu_available = True
        except Exception:
            pass

    return snap


def snapshot_delta(before: ResourceSnapshot, after: ResourceSnapshot) -> dict:
    """Compute deltas between two snapshots (disk I/O, CPU time, RAM change)."""
    def _diff(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 3)

    return {
        "cpu_time_user_delta": _diff(before.cpu_time_user, after.cpu_time_user),
        "cpu_time_sys_delta": _diff(before.cpu_time_sys, after.cpu_time_sys),
        "ram_delta_mb": _diff(before.ram_mb, after.ram_mb),
        "disk_read_delta_mb": _diff(before.disk_read_mb, after.disk_read_mb),
        "disk_write_delta_mb": _diff(before.disk_write_mb, after.disk_write_mb),
        "gpu_mem_delta_mb": _diff(before.gpu_mem_used_mb, after.gpu_mem_used_mb),
    }


# ─── Background monitor (for long ingestion jobs) ─────────────────────────────

class BackgroundResourceMonitor:
    """
    Daemon thread that samples resources every ``interval_s`` seconds during a
    long-running job. Tracks peak RAM, peak GPU memory, and peak CPU%.

    Usage:
        monitor = BackgroundResourceMonitor()
        monitor.start()
        # ... run long job ...
        monitor.stop()
        summary = monitor.get_summary()
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._samples: list[ResourceSnapshot] = []
        self.start_snapshot: Optional[ResourceSnapshot] = None
        self.end_snapshot: Optional[ResourceSnapshot] = None

    def start(self):
        self.start_snapshot = take_snapshot()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ResourceMonitor")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.end_snapshot = take_snapshot()

    def _run(self):
        while not self._stop_event.wait(timeout=self.interval_s):
            snap = take_snapshot()
            with self._lock:
                self._samples.append(snap)

    def get_summary(self) -> dict:
        """Return peak/average resource usage since start()."""
        with self._lock:
            samples = list(self._samples)

        if not samples:
            end = self.end_snapshot or take_snapshot()
            return {
                "cpu_peak_pct": end.cpu_percent,
                "ram_peak_mb": end.ram_mb,
                "gpu_peak_mem_mb": end.gpu_mem_used_mb,
                "gpu_peak_util_pct": end.gpu_util_pct,
                "sample_count": 0,
            }

        def _peak(attr):
            vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
            return round(max(vals), 2) if vals else None

        def _avg(attr):
            vals = [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        start = self.start_snapshot
        end = self.end_snapshot or (samples[-1] if samples else None)

        delta = snapshot_delta(start, end) if start and end else {}

        return {
            "cpu_peak_pct": _peak("cpu_percent"),
            "cpu_avg_pct": _avg("cpu_percent"),
            "ram_start_mb": round(start.ram_mb, 2) if start and start.ram_mb else None,
            "ram_peak_mb": _peak("ram_mb"),
            "ram_end_mb": round(end.ram_mb, 2) if end and end.ram_mb else None,
            "gpu_available": any(s.gpu_available for s in samples),
            "gpu_peak_mem_mb": _peak("gpu_mem_used_mb"),
            "gpu_peak_util_pct": _peak("gpu_util_pct"),
            "disk_read_delta_mb": delta.get("disk_read_delta_mb"),
            "disk_write_delta_mb": delta.get("disk_write_delta_mb"),
            "cpu_time_user_delta": delta.get("cpu_time_user_delta"),
            "cpu_time_sys_delta": delta.get("cpu_time_sys_delta"),
            "sample_count": len(samples),
        }
