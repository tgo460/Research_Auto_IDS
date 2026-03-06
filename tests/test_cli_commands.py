import json
import os
import subprocess
import sys


def test_validate_ids_cli():
    if not os.path.exists("configs/deployment.example.json"):
        return
    cmd = [sys.executable, "validate_ids.py", "--config", "configs/deployment.example.json"]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert '"valid": true' in res.stdout.lower()


def test_benchmark_ids_schema(tmp_path):
    if not os.path.exists("configs/deployment.example.json"):
        return
    out = tmp_path / "bench.json"
    cmd = [
        sys.executable,
        "benchmark_ids.py",
        "--config",
        "configs/deployment.example.json",
        "--output",
        str(out),
        "--max_samples",
        "2",
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    rep = data["benchmark_report"]
    for key in (
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "fpr",
        "fnr",
        "mcc",
        "cpu_percent",
        "memory_mb",
        "power_watts",
        "hardware_id",
        "os",
        "model_hash",
    ):
        assert key in rep
