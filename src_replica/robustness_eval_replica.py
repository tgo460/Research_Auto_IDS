import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.realtime_inference_replica import RealTimeEngine
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD


def evaluate_noise(engine: RealTimeEngine, ds: CorrelatedHybridVehicleDataset, sigma: float, max_samples: int):
    y_true = []
    y_pred = []
    for i in range(min(max_samples, len(ds))):
        sample = ds[i]
        xc = sample["can"].clone()
        xe = sample["eth"].clone()
        label = int(sample["label"].item())
        if sigma > 0:
            xc = xc + torch.randn_like(xc) * sigma
            xe = xe + torch.randn_like(xe) * sigma
        out = engine.process_packet(xc, xe)
        y_true.append(label)
        y_pred.append(int(out["prediction"]))
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return {"sigma": sigma, "accuracy": acc}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robustness/noise evaluation on runtime engine.")
    parser.add_argument("--light_model", default="models/student_tiny_improved.pth")
    parser.add_argument("--heavy_model", default="models/heavy_rf_improved.joblib")
    parser.add_argument("--onnx_model", default="models/student_tiny_improved.onnx")
    parser.add_argument("--can_csv", default="datasets/replica_can_b1_engineered/can_dos_train.csv")
    parser.add_argument(
        "--eth_csv",
        default="datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv",
    )
    parser.add_argument("--eth_npy", default="datasets/eth_driving_01_injected_images-003.npy")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--output", default="reports/robustness_report.json")
    args = parser.parse_args()

    can_features = [
        "CAN_ID",
        "DLC",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "can_id_freq_global",
        "can_id_freq_win",
        "payload_entropy",
        "inter_arrival",
        "inter_arrival_roll_mean",
        "id_switch_rate_win",
    ]

    ds = CorrelatedHybridVehicleDataset(
        can_csv_path=args.can_csv,
        eth_packet_csv_path=args.eth_csv,
        eth_npy_path=args.eth_npy,
        can_features=can_features,
        can_window_size=CAN_WINDOW_SIZE_STANDARD,
        eth_window_size=ETH_WINDOW_SIZE_STANDARD,
        eth_overlap=0,
    )
    engine = RealTimeEngine(
        light_model_path=args.light_model,
        heavy_model_path=args.heavy_model,
        threshold=0.5781,
        input_dim=16,
        onnx_model_path=args.onnx_model,
        use_onnx=True,
    )

    rows = []
    for sigma in (0.0, 0.01, 0.05, 0.1):
        rows.append(evaluate_noise(engine, ds, sigma=sigma, max_samples=args.max_samples))
    base_acc = rows[0]["accuracy"] if rows else 0.0
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type": "robustness_eval",
        "rows": rows,
        "degradation_vs_clean": [
            {"sigma": r["sigma"], "delta_accuracy": r["accuracy"] - base_acc} for r in rows
        ],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
