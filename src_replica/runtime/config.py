import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .standards import enforce_window_standard


@dataclass
class DeploymentConfig:
    light_model_path: str
    heavy_model_path: str
    onnx_model_path: str
    can_window_size: int
    eth_window_size: int
    routing_threshold: float
    latency_budget_ms: float
    fpr_budget: float
    fail_safe_mode: str
    model_integrity_sha256: str
    can_source: str = ""
    eth_source: str = ""
    alert_output_path: str = "logs/runtime_alerts.jsonl"
    use_onnx: bool = True
    max_samples: int = 100

    def validate(self) -> None:
        required = [
            "light_model_path",
            "heavy_model_path",
            "onnx_model_path",
            "model_integrity_sha256",
            "fail_safe_mode",
        ]
        for key in required:
            value = getattr(self, key)
            if not value:
                raise ValueError(f"missing required config: {key}")
        if self.fail_safe_mode not in {"fail_open", "fail_closed", "degraded"}:
            raise ValueError("fail_safe_mode must be one of fail_open, fail_closed, degraded")
        enforce_window_standard(self.can_window_size, self.eth_window_size)
        if not (0.0 < self.routing_threshold < 1.0):
            raise ValueError("routing_threshold must be in (0,1)")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be > 0")
        if not (0.0 < self.fpr_budget < 1.0):
            raise ValueError("fpr_budget must be in (0,1)")
        for path in (self.light_model_path, self.heavy_model_path, self.onnx_model_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_deployment_config(path: str) -> DeploymentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = DeploymentConfig(**raw)
    cfg.validate()
    return cfg
