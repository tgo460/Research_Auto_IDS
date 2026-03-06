import argparse
import json
import os
from datetime import datetime, timezone
from typing import Tuple

import numpy as np

from src_replica.runtime.adapters import (
    CanAlertEgress,
    CsvCanIngest,
    CsvEthIngest,
    FileAlertEgress,
    PcapEthIngest,
    SocketCanIngest,
    StdoutAlertEgress,
    SomeIPEgress,
    WatchdogHealthMonitor,
)
from src_replica.runtime.config import DeploymentConfig, load_deployment_config
from src_replica.runtime.engine import RuntimeIDSService
from src_replica.runtime.metrics import make_benchmark_report
from src_replica.runtime.security import sha256_file


def _load_or_override_config(args) -> DeploymentConfig:
    cfg = load_deployment_config(args.config)
    if getattr(args, "can_csv", None):
        cfg.can_source = args.can_csv
    if getattr(args, "eth_csv", None):
        cfg.eth_source = args.eth_csv
    if getattr(args, "max_samples", None) is not None:
        cfg.max_samples = int(args.max_samples)
    return cfg


def _make_service(cfg: DeploymentConfig, args, alert_to_stdout: bool = False) -> RuntimeIDSService:
    if args.can_mode == "socketcan":
        can_ingest = SocketCanIngest(channel=args.can_channel, bustype=args.can_bustype)
    else:
        can_ingest = CsvCanIngest(cfg.can_source)

    if args.eth_mode == "pcap":
        eth_ingest = PcapEthIngest(cfg.eth_source)
    else:
        eth_ingest = CsvEthIngest(cfg.eth_source)

    if alert_to_stdout:
        egress = StdoutAlertEgress()
    elif args.egress_mode == "can":
        egress = CanAlertEgress(
            channel=args.egress_can_channel,
            bustype=args.egress_can_bustype,
            arbitration_id=args.egress_can_id,
        )
    elif args.egress_mode == "someip":
        egress = SomeIPEgress(host=args.someip_host, port=args.someip_port)
    else:
        egress = FileAlertEgress(cfg.alert_output_path)

    health = WatchdogHealthMonitor(latency_budget_ms=cfg.latency_budget_ms)
    return RuntimeIDSService(cfg, can_ingest, eth_ingest, egress, health)


def run_ids(args) -> int:
    cfg = _load_or_override_config(args)
    svc = _make_service(cfg, args, alert_to_stdout=args.stdout_alerts)
    y_true, y_pred, lat = svc.run(max_samples=cfg.max_samples)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(y_true),
        "attack_predictions": int(np.sum(np.asarray(y_pred) == 1)) if y_pred else 0,
        "avg_latency_ms": float(np.mean(lat)) if lat else 0.0,
        "alerts_path": cfg.alert_output_path if not args.stdout_alerts else "stdout",
    }
    print(json.dumps(summary, indent=2))
    return 0


def benchmark_ids(args) -> int:
    cfg = _load_or_override_config(args)
    svc = _make_service(cfg, args, alert_to_stdout=False)
    y_true, y_pred, lat = svc.run(max_samples=cfg.max_samples)
    report = make_benchmark_report(
        y_true=np.asarray(y_true, dtype=int),
        y_pred=np.asarray(y_pred, dtype=int),
        latencies_ms=np.asarray(lat, dtype=float),
        model_hash=svc.model_hash(),
        deadline_ms=cfg.latency_budget_ms,
    ).to_dict()
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "benchmark_ids",
        "report_schema_version": "1.0",
        "deployment_config": cfg.to_dict(),
        "benchmark_report": report,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved benchmark report: {args.output}")
    return 0


def validate_ids(args) -> int:
    cfg = load_deployment_config(args.config)
    model_hash = sha256_file(cfg.onnx_model_path)
    if model_hash.lower() != cfg.model_integrity_sha256.lower():
        raise ValueError("model_integrity_sha256 mismatch")

    import onnxruntime as ort

    sess = ort.InferenceSession(cfg.onnx_model_path)
    inputs = sess.get_inputs()
    if len(inputs) != 2:
        raise ValueError("ONNX model must expose exactly two inputs: CAN and ETH")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "validate_ids",
        "valid": True,
        "config": args.config,
        "onnx_inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in inputs
        ],
        "model_hash": model_hash,
    }
    print(json.dumps(report, indent=2))
    return 0


def replay_can_eth(args) -> int:
    cfg = _load_or_override_config(args)
    svc = _make_service(cfg, args, alert_to_stdout=args.stdout_alerts)
    y_true, y_pred, lat = svc.run(max_samples=cfg.max_samples)
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "replay_can_eth",
        "samples": len(y_true),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0,
        "mean_latency_ms": float(np.mean(lat)) if lat else 0.0,
        "can_csv": cfg.can_source,
        "eth_csv": cfg.eth_source,
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automotive IDS runtime CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_runtime_options(sp):
        sp.add_argument("--can-mode", choices=["csv", "socketcan"], default="csv")
        sp.add_argument("--eth-mode", choices=["csv", "pcap"], default="csv")
        sp.add_argument("--can-channel", default="can0")
        sp.add_argument("--can-bustype", default="socketcan")
        sp.add_argument("--egress-mode", choices=["file", "stdout", "can", "someip"], default="file")
        sp.add_argument("--egress-can-channel", default="can0")
        sp.add_argument("--egress-can-bustype", default="socketcan")
        sp.add_argument("--egress-can-id", type=lambda x: int(x, 0), default=0x6A0)
        sp.add_argument("--someip-host", default="127.0.0.1")
        sp.add_argument("--someip-port", type=int, default=30490)

    p_run = sub.add_parser("run_ids", help="Run IDS inference from configured sources")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--max_samples", type=int, default=None)
    p_run.add_argument("--stdout-alerts", action="store_true")
    add_runtime_options(p_run)
    p_run.set_defaults(func=run_ids)

    p_bm = sub.add_parser("benchmark_ids", help="Benchmark IDS and write strict report schema")
    p_bm.add_argument("--config", required=True)
    p_bm.add_argument("--output", default="reports/benchmark_ids_report.json")
    p_bm.add_argument("--max_samples", type=int, default=None)
    add_runtime_options(p_bm)
    p_bm.set_defaults(func=benchmark_ids)

    p_val = sub.add_parser("validate_ids", help="Validate deployment config + ONNX contract")
    p_val.add_argument("--config", required=True)
    p_val.set_defaults(func=validate_ids)

    p_rep = sub.add_parser("replay_can_eth", help="Replay CAN+ETH CSV sources through IDS")
    p_rep.add_argument("--config", required=True)
    p_rep.add_argument("--can_csv", required=True)
    p_rep.add_argument("--eth_csv", required=True)
    p_rep.add_argument("--max_samples", type=int, default=None)
    p_rep.add_argument("--stdout-alerts", action="store_true")
    add_runtime_options(p_rep)
    p_rep.set_defaults(func=replay_can_eth)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
