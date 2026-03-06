import json
import os
import subprocess
import sys


def test_coordinated_attack_replay_report(tmp_path):
    if not os.path.exists("configs/deployment.example.json"):
        return
    out = tmp_path / "coord.json"
    cmd = [
        sys.executable,
        "src_replica/coordinated_attack_replay.py",
        "--config",
        "configs/deployment.example.json",
        "--max_samples",
        "10",
        "--output",
        str(out),
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "detection_rate" in data
    assert "fpr" in data
