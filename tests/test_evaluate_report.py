import json
import os
import subprocess
import sys


def test_evaluate_outputs_mcc_fnr(tmp_path):
    cmd = [
        sys.executable,
        "evaluate.py",
        "--base-path",
        ".",
        "--out-dir",
        str(tmp_path),
        "--strict-split-check",
        "--split-manifest",
        "data/splits/split_v2_domain_balanced.json",
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = tmp_path / "final_metrics_latest.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "fnr" in data["ids_metrics"]["light_only"]
    assert "mcc" in data["ids_metrics"]["light_only"]
    assert "fnr" in data["ids_metrics"]["cascade"]
    assert "mcc" in data["ids_metrics"]["cascade"]
