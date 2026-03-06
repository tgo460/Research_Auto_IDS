import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.realtime_inference_replica import RealTimeEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate runtime contract and malformed-input rejection.")
    parser.add_argument("--light_model", default="models/student_tiny_improved.pth")
    parser.add_argument("--heavy_model", default="models/heavy_rf_improved.joblib")
    parser.add_argument("--onnx_model", default="models/student_tiny_improved.onnx")
    parser.add_argument("--output", default="reports/runtime_validation_report.json")
    args = parser.parse_args()

    engine = RealTimeEngine(
        light_model_path=args.light_model,
        heavy_model_path=args.heavy_model,
        threshold=0.5781,
        input_dim=16,
        onnx_model_path=args.onnx_model,
        use_onnx=True,
    )

    passed = []
    rejected = []
    # Valid input
    can_ok = torch.randn(100, 16)
    eth_ok = torch.randn(1, 1, 32, 32)
    _ = engine.process_packet(can_ok, eth_ok)
    passed.append("valid_shape")

    bad_cases = [
        ("bad_can_rank", torch.randn(16), eth_ok),
        ("bad_can_window", torch.randn(32, 16), eth_ok),
        ("bad_eth_rank", can_ok, torch.randn(32, 32)),
    ]
    for name, can_x, eth_x in bad_cases:
        try:
            _ = engine.process_packet(can_x, eth_x)
            passed.append(name)
        except Exception as exc:
            rejected.append({"case": name, "error": str(exc)})

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_contract_valid": True,
        "malformed_rejections": rejected,
        "unexpected_pass_cases": [x for x in passed if x != "valid_shape"],
        "all_malformed_rejected": len(rejected) == len(bad_cases),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
