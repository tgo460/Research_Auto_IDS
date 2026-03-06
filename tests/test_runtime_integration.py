import json
import os

from src_replica.runtime.adapters import (
    CsvCanIngest,
    CsvEthIngest,
    FileAlertEgress,
    WatchdogHealthMonitor,
)
from src_replica.runtime.config import DeploymentConfig
from src_replica.runtime.engine import RuntimeIDSService
from src_replica.runtime.security import sha256_file


def test_runtime_replay_generates_alerts(tmp_path):
    onnx = "models/student_tiny_improved.onnx"
    if not os.path.exists(onnx):
        return

    alerts_path = tmp_path / "alerts.jsonl"
    cfg = DeploymentConfig(
        light_model_path="models/student_tiny_improved.pth",
        heavy_model_path="models/heavy_rf_improved.joblib",
        onnx_model_path=onnx,
        can_window_size=100,
        eth_window_size=1,
        routing_threshold=0.5781,
        latency_budget_ms=100.0,
        fpr_budget=0.05,
        fail_safe_mode="degraded",
        model_integrity_sha256=sha256_file(onnx),
        can_source="datasets/can_dos_train.csv",
        eth_source="datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv",
        alert_output_path=str(alerts_path),
        use_onnx=True,
        max_samples=2,
    )
    cfg.validate()

    svc = RuntimeIDSService(
        cfg=cfg,
        can_ingest=CsvCanIngest(cfg.can_source),
        eth_ingest=CsvEthIngest(cfg.eth_source),
        alert_egress=FileAlertEgress(cfg.alert_output_path),
        health_monitor=WatchdogHealthMonitor(latency_budget_ms=cfg.latency_budget_ms),
    )
    y_true, y_pred, lat = svc.run(max_samples=2)
    assert len(y_true) == len(y_pred) == len(lat) == 2
    with open(alerts_path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == 2
    required = {
        "timestamp",
        "source",
        "attack_class",
        "confidence",
        "route_path",
        "latency_ms",
        "decision_id",
    }
    assert required.issubset(lines[0].keys())
