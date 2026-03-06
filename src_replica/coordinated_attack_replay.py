import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.runtime.adapters import CsvCanIngest, CsvEthIngest, FileAlertEgress, WatchdogHealthMonitor
from src_replica.runtime.config import load_deployment_config
from src_replica.runtime.engine import RuntimeIDSService


class LabeledCsvEthIngest(CsvEthIngest):
    def __init__(self, csv_path: str, attack_stride: int = 50):
        super().__init__(csv_path)
        self.attack_stride = max(1, int(attack_stride))

    def read_frame(self):
        frame = super().read_frame()
        if frame is None:
            return None
        # Synthetic coordinated trigger for replay testing.
        frame_idx = self._idx
        frame["label"] = 1 if (frame_idx % self.attack_stride == 0) else 0
        return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay synthetic coordinated Ethernet->CAN attack scenarios.")
    parser.add_argument("--config", default="configs/deployment.example.json")
    parser.add_argument("--can_csv", default="datasets/can_dos_train.csv")
    parser.add_argument(
        "--eth_csv",
        default="datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv",
    )
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--attack_stride", type=int, default=50)
    parser.add_argument("--output", default="reports/coordinated_attack_report.json")
    args = parser.parse_args()

    cfg = load_deployment_config(args.config)
    cfg.can_source = args.can_csv
    cfg.eth_source = args.eth_csv
    cfg.max_samples = args.max_samples

    svc = RuntimeIDSService(
        cfg=cfg,
        can_ingest=CsvCanIngest(cfg.can_source),
        eth_ingest=LabeledCsvEthIngest(cfg.eth_source, attack_stride=args.attack_stride),
        alert_egress=FileAlertEgress(cfg.alert_output_path),
        health_monitor=WatchdogHealthMonitor(latency_budget_ms=cfg.latency_budget_ms),
    )
    y_true, y_pred, lat = svc.run(max_samples=cfg.max_samples)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    dr = float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0
    fpr = float(np.mean(y_pred[y_true == 0] == 1)) if np.any(y_true == 0) else 0.0
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type": "coordinated_attack_replay",
        "samples": int(len(y_true)),
        "attack_samples": int(np.sum(y_true == 1)),
        "normal_samples": int(np.sum(y_true == 0)),
        "detection_rate": dr,
        "fpr": fpr,
        "latency_ms": {
            "mean": float(np.mean(lat)) if lat else 0.0,
            "p95": float(np.percentile(lat, 95)) if lat else 0.0,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
