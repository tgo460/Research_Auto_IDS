import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort


def _latency_ms(session: ort.InferenceSession, n: int = 100) -> float:
    inputs = session.get_inputs()
    can_shape = [1 if not isinstance(d, int) else d for d in inputs[0].shape]
    eth_shape = [1 if not isinstance(d, int) else d for d in inputs[1].shape]
    can = np.random.randn(*can_shape).astype(np.float32)
    eth = np.random.randn(*eth_shape).astype(np.float32)
    feed = {inputs[0].name: can, inputs[1].name: eth}
    for _ in range(10):
        session.run(None, feed)
    t0 = time.time()
    for _ in range(n):
        session.run(None, feed)
    t1 = time.time()
    return (t1 - t0) * 1000.0 / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to int8 and benchmark delta.")
    parser.add_argument("--input", default="models/student_tiny_improved.onnx")
    parser.add_argument("--output", default="models/student_tiny_improved.int8.onnx")
    parser.add_argument("--report", default="reports/quantization_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_model": args.input,
            "quantized_model": None,
            "status": "skipped",
            "reason": f"quantization dependencies unavailable: {exc}",
        }
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return 0

    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QInt8,
    )

    s_fp = ort.InferenceSession(args.input)
    s_q = ort.InferenceSession(args.output)
    fp_lat = _latency_ms(s_fp)
    q_lat = _latency_ms(s_q)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_model": args.input,
        "quantized_model": args.output,
        "latency_ms": {"fp32": fp_lat, "int8": q_lat},
        "speedup": (fp_lat / q_lat) if q_lat > 0 else 0.0,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
