import json
import os

import pytest

from src_replica.runtime.adapters import WatchdogHealthMonitor
from src_replica.runtime.config import DeploymentConfig
from src_replica.runtime.security import sha256_file, verify_model_hash


def _valid_cfg() -> DeploymentConfig:
    return DeploymentConfig(
        light_model_path="models/student_tiny_improved.pth",
        heavy_model_path="models/heavy_rf_improved.joblib",
        onnx_model_path="models/student_tiny_improved.onnx",
        can_window_size=100,
        eth_window_size=1,
        routing_threshold=0.55,
        latency_budget_ms=100.0,
        fpr_budget=0.05,
        fail_safe_mode="degraded",
        model_integrity_sha256=sha256_file("models/student_tiny_improved.onnx"),
        can_source="datasets/can_dos_train.csv",
        eth_source="datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv",
        max_samples=5,
    )


def test_window_standard_enforced():
    cfg = _valid_cfg()
    cfg.can_window_size = 10
    with pytest.raises(ValueError):
        cfg.validate()


def test_hash_integrity_verification():
    path = "models/student_tiny_improved.onnx"
    if not os.path.exists(path):
        return
    h = sha256_file(path)
    verify_model_hash(path, h)
    with pytest.raises(ValueError):
        verify_model_hash(path, "0" * 64)


def test_watchdog_trips_on_deadline_miss():
    wd = WatchdogHealthMonitor(latency_budget_ms=1.0, max_consecutive_misses=2)
    wd.heartbeat(5.0)
    assert wd.tripped() is False
    wd.heartbeat(6.0)
    assert wd.tripped() is True
    assert wd.last_reason() != ""


def test_deployment_example_is_valid_schema_instance():
    cfg_path = "configs/deployment.example.json"
    if not os.path.exists(cfg_path):
        pytest.skip("missing deployment example")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = DeploymentConfig(**raw)
    cfg.validate()
