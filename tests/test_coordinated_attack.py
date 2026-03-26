import json
import os
import subprocess
import sys

import pandas as pd


def test_coordinated_attack_replay_report(tmp_path):
    if not os.path.exists("configs/deployment.example.json"):
        return

    can_dir = tmp_path / "datasets"
    eth_dir = can_dir / "replica_eth_smoke"
    can_dir.mkdir()
    eth_dir.mkdir(parents=True)

    def write_can(path, label):
        rows = []
        for idx in range(110):
            rows.append(
                {
                    "Timestamp": idx * 0.001,
                    "CAN_ID": 0.1 if idx % 2 == 0 else 0.2,
                    "DLC": 1.0,
                    "D0": 0.0,
                    "D1": 0.0,
                    "D2": 0.0,
                    "D3": 0.0,
                    "D4": 0.0,
                    "D5": 0.0,
                    "D6": 0.0,
                    "D7": 0.0,
                    "Label": label if idx >= 100 else 0,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    def write_eth(path, label):
        rows = []
        for idx in range(20):
            rows.append(
                {
                    "packet_index": idx,
                    "timestamp_sec": 1,
                    "timestamp_usec": idx * 1000,
                    "captured_len": 64 + idx,
                    "original_len": 64 + idx,
                    "Label": label,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    write_can(can_dir / "can_normal_train.csv", 0)
    write_can(can_dir / "can_dos_train.csv", 1)
    write_eth(eth_dir / "eth_driving_01_original_replica_packets.csv", 0)
    write_eth(eth_dir / "eth_driving_01_injected_replica_packets.csv", 1)

    out = tmp_path / "coord.json"
    cmd = [
        sys.executable,
        "src_replica/coordinated_attack_replay.py",
        "--config",
        "configs/deployment.example.json",
        "--can_dir",
        str(can_dir),
        "--eth_dir",
        str(eth_dir),
        "--max_samples",
        "2",
        "--output",
        str(out),
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "overall_detection_rate" in data
    assert "overall_false_positive_rate" in data
    assert "per_scenario" in data
    eth_only = [row for row in data["per_scenario"] if row["name"].startswith("eth_only_")]
    assert eth_only
    assert any(row.get("attack_samples", 0) > 0 for row in eth_only)
