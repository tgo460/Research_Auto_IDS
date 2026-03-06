import platform
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import matthews_corrcoef

from .contracts import BenchmarkReport


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def fpr_from_counts(c: Dict[str, int]) -> float:
    denom = c["fp"] + c["tn"]
    return 0.0 if denom == 0 else float(c["fp"] / denom)


def fnr_from_counts(c: Dict[str, int]) -> float:
    denom = c["fn"] + c["tp"]
    return 0.0 if denom == 0 else float(c["fn"] / denom)


def compute_cpu_memory() -> Dict[str, float]:
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=0.05)),
            "memory_mb": float(p.memory_info().rss / (1024 * 1024)),
        }
    except Exception:
        return {"cpu_percent": 0.0, "memory_mb": 0.0}


def compute_power_watts() -> Optional[float]:
    # Portable power telemetry is platform-specific; keep nullable contract field.
    return None


def make_benchmark_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latencies_ms: np.ndarray,
    model_hash: str,
    deadline_ms: float,
) -> BenchmarkReport:
    counts = confusion_counts(y_true, y_pred)
    fpr = fpr_from_counts(counts)
    fnr = fnr_from_counts(counts)
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(y_true) else 0.0
    sys_stats = compute_cpu_memory()
    misses = float(np.mean(latencies_ms > deadline_ms)) if len(latencies_ms) else 0.0
    return BenchmarkReport(
        latency_p50_ms=float(np.percentile(latencies_ms, 50)) if len(latencies_ms) else 0.0,
        latency_p95_ms=float(np.percentile(latencies_ms, 95)) if len(latencies_ms) else 0.0,
        latency_max_ms=float(np.max(latencies_ms)) if len(latencies_ms) else 0.0,
        fpr=float(fpr),
        fnr=float(fnr),
        mcc=float(mcc),
        cpu_percent=float(sys_stats["cpu_percent"]),
        memory_mb=float(sys_stats["memory_mb"]),
        power_watts=compute_power_watts(),
        hardware_id=platform.machine(),
        os=platform.platform(),
        model_hash=model_hash,
        total_samples=int(len(y_true)),
        deadline_ms=float(deadline_ms),
        deadline_miss_rate=float(misses),
    )
