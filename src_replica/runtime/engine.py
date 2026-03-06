import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src_replica.realtime_inference_replica import RealTimeEngine

from .adapters import (
    AlertEgress,
    CanIngest,
    EthIngest,
    HealthMonitor,
    to_can_window,
    to_eth_image,
)
from .config import DeploymentConfig
from .contracts import AlertRecord
from .security import sha256_file, verify_model_hash


class RuntimeIDSService:
    def __init__(
        self,
        cfg: DeploymentConfig,
        can_ingest: CanIngest,
        eth_ingest: EthIngest,
        alert_egress: AlertEgress,
        health_monitor: HealthMonitor,
    ):
        self.cfg = cfg
        self.can_ingest = can_ingest
        self.eth_ingest = eth_ingest
        self.alert_egress = alert_egress
        self.health_monitor = health_monitor
        verify_model_hash(cfg.onnx_model_path, cfg.model_integrity_sha256)
        input_dim = 16 if "improved" in cfg.light_model_path.lower() else 10
        self.engine = RealTimeEngine(
            light_model_path=cfg.light_model_path,
            heavy_model_path=cfg.heavy_model_path,
            threshold=cfg.routing_threshold,
            input_dim=input_dim,
            onnx_model_path=cfg.onnx_model_path,
            use_onnx=cfg.use_onnx,
        )
        self.input_dim = input_dim
        self.can_buf: deque = deque(maxlen=cfg.can_window_size)
        self.last_eth: Optional[Dict] = None

    def _prepare_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        can_np = to_can_window(self.can_buf, self.cfg.can_window_size)
        if can_np.shape[1] < self.input_dim:
            pad_width = self.input_dim - can_np.shape[1]
            can_np = np.pad(can_np, ((0, 0), (0, pad_width)), mode="constant")
        elif can_np.shape[1] > self.input_dim:
            can_np = can_np[:, : self.input_dim]

        eth_np = to_eth_image(self.last_eth or {})
        can_t = torch.tensor(can_np, dtype=torch.float32)
        eth_t = torch.tensor(eth_np, dtype=torch.float32).unsqueeze(0)
        return can_t, eth_t

    def _apply_fail_safe(self, result: Dict) -> Dict:
        mode = self.cfg.fail_safe_mode
        if mode == "fail_open":
            result["prediction"] = 0
            result["source"] = "FAIL_OPEN"
        elif mode == "fail_closed":
            result["prediction"] = 1
            result["source"] = "FAIL_CLOSED"
        elif mode == "degraded":
            result["source"] = "DEGRADED"
        return result

    def process_once(self) -> Optional[Dict]:
        can_fr = self.can_ingest.read_frame()
        eth_fr = self.eth_ingest.read_frame()
        if can_fr is not None:
            self.can_buf.append(can_fr)
        if eth_fr is not None:
            self.last_eth = eth_fr

        if len(self.can_buf) < self.cfg.can_window_size or self.last_eth is None:
            return None

        can_t, eth_t = self._prepare_inputs()
        result = self.engine.process_packet(can_t, eth_t)
        self.health_monitor.heartbeat(float(result["latency_ms"]))
        if self.health_monitor.tripped():
            result = self._apply_fail_safe(result)
            result["health_reason"] = self.health_monitor.last_reason()
        return result

    def run(self, max_samples: Optional[int] = None) -> Tuple[List[int], List[int], List[float]]:
        n = 0
        idle_loops = 0
        y_true: List[int] = []
        y_pred: List[int] = []
        latencies: List[float] = []
        limit = max_samples if max_samples is not None else self.cfg.max_samples
        while n < limit:
            out = self.process_once()
            if out is None:
                idle_loops += 1
                if idle_loops > (self.cfg.can_window_size * 4):
                    break
                continue
            idle_loops = 0

            pred = int(out["prediction"])
            label = int(max(self.can_buf[-1].get("label", 0), self.last_eth.get("label", 0)))
            y_true.append(label)
            y_pred.append(pred)
            latencies.append(float(out["latency_ms"]))
            alert = AlertRecord.now(
                source="IDS",
                attack_class="MALICIOUS" if pred == 1 else "NORMAL",
                confidence=float(out.get("confidence", 0.0)),
                route_path=str(out.get("source", "LIGHT")),
                latency_ms=float(out.get("latency_ms", 0.0)),
                decision_id=str(uuid.uuid4()),
                details={
                    "label": label,
                    "heavy_overhead_ms": float(out.get("heavy_overhead_ms", 0.0)),
                },
            ).to_dict()
            self.alert_egress.publish(alert)
            n += 1

        return y_true, y_pred, latencies

    def model_hash(self) -> str:
        return sha256_file(self.cfg.onnx_model_path)
