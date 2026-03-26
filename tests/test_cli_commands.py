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

    eth_csv = tmp_path / "eth_labeled_replica_packets.csv"
    eth_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len,Label\n"
        "0,1,0,64,64,1\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "deployment.json"
    cfg = json.loads(open("configs/deployment.example.json", "r", encoding="utf-8").read())
    cfg["eth_source"] = str(eth_csv)
    cfg["heavy_model_path"] = "models/heavy_rf_improved.joblib"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    out = tmp_path / "bench.json"
    cmd = [
        sys.executable,
        "benchmark_ids.py",
        "--config",
        str(cfg_path),
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


def test_benchmark_ids_rejects_pcap_mode(tmp_path):
    if not os.path.exists("configs/deployment.example.json"):
        return

    cfg_path = tmp_path / "deployment.json"
    with open("configs/deployment.example.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["heavy_model_path"] = "models/heavy_rf_improved.joblib"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    cmd = [
        sys.executable,
        "benchmark_ids.py",
        "--config",
        str(cfg_path),
        "--eth-mode",
        "pcap",
        "--max_samples",
        "1",
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode != 0
    assert "requires labeled Ethernet replay data" in (res.stderr + res.stdout)
