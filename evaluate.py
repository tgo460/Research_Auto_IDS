import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fnr_from_cm(cm: List[List[int]]) -> float:
    tn, fp = cm[0]
    fn, tp = cm[1]
    denom = fn + tp
    return 0.0 if denom == 0 else float(fn / denom)


def _mcc_from_cm(cm: List[List[int]]) -> float:
    tn, fp = [float(x) for x in cm[0]]
    fn, tp = [float(x) for x in cm[1]]
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0:
        return 0.0
    return float(((tp * tn) - (fp * fn)) / denom)


def _validate_split_manifest(path: str, datasets_dir: str) -> Dict:
    data = _load_json(path)
    errors = []
    stats = {"eth": {}, "can": {}}
    for modality in ("eth", "can"):
        for split in ("train", "val", "test"):
            files = data.get("modalities", {}).get(modality, {}).get(split, [])
            stats[modality][split] = len(files)
            for rel in files:
                full = os.path.join(datasets_dir, rel)
                if not os.path.exists(full):
                    errors.append(f"missing {modality}/{split} file: {rel}")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": path.replace("\\", "/"),
        "datasets_dir": datasets_dir.replace("\\", "/"),
        "ok": len(errors) == 0,
        "errors": errors,
        "stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate IDS metrics into final report")
    parser.add_argument("--base-path", default=".")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--strict-split-check", action="store_true")
    parser.add_argument("--split-manifest", default="data/splits/split_v2_domain_balanced.json")
    args = parser.parse_args()

    base = args.base_path
    logs_dir = os.path.join(base, "logs")
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    edge_path = os.path.join(logs_dir, "edge_benchmark_report_latest.json")
    router_path = os.path.join(logs_dir, "router_eval_report_latest.json")
    cascade_path = os.path.join(logs_dir, "cascade_eval_report_latest.json")
    corr_path = os.path.join(logs_dir, "correlation_report_latest.json")

    edge = _load_json(edge_path)
    router = _load_json(router_path)
    cascade = _load_json(cascade_path)
    corr = _load_json(corr_path)

    cm_light = cascade.get("confusion_matrix", {}).get("light_only", [[0, 0], [0, 0]])
    cm_cascade = cascade.get("confusion_matrix", {}).get("cascade", [[0, 0], [0, 0]])

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "edge_benchmark_report": edge_path.replace("\\", "/"),
            "router_eval_report": router_path.replace("\\", "/"),
            "cascade_eval_report": cascade_path.replace("\\", "/"),
            "correlation_report": corr_path.replace("\\", "/"),
        },
        "ids_metrics": {
            "light_only": {
                **cascade.get("light_only", {}),
                "fnr": _fnr_from_cm(cm_light),
                "mcc": _mcc_from_cm(cm_light),
            },
            "cascade": {
                **cascade.get("cascade", {}),
                "fnr": _fnr_from_cm(cm_cascade),
                "mcc": _mcc_from_cm(cm_cascade),
            },
            "pr_auc": cascade.get("pr_auc", {}),
            "improvement": cascade.get("improvement", {}),
            "fpr_budget": cascade.get("fpr_budget"),
            "fpr_budget_policy": cascade.get("fpr_budget_policy"),
            "detection_delay_ms_proxy": {
                "median": corr.get("alignment", {}).get("median_delta_ms", 0.0),
                "p95": corr.get("alignment", {}).get("p95_delta_ms", 0.0),
            },
            "route_analytics": {
                "routed_fraction": router.get("actual_routed_fraction", 0.0),
                "light_samples": router.get("samples", {}).get("light_path", 0),
                "heavy_samples": router.get("samples", {}).get("heavy_path", 0),
            },
            "confusion_matrices": {
                "overall_cascade": cm_cascade,
                "light_path": router.get("confusion_matrix", {}).get("light_path", [[0, 0], [0, 0]]),
                "heavy_path": router.get("confusion_matrix", {}).get("heavy_path", [[0, 0], [0, 0]]),
            },
        },
        "deployment_metrics": {
            "model_size_kb": edge.get("model_size_kb"),
            "torchscript_size_kb": edge.get("torchscript_size_kb"),
            "latency_ms": edge.get("latency", {}),
            "throughput_samples_per_sec": edge.get("latency", {}).get(
                "throughput_samples_per_sec", 0.0
            ),
            "environment": edge.get("environment", {}),
        },
        "artifacts": {
            "plots": {
                "cascade_confusion": "reports/confusion_cascade.png",
                "light_confusion": "reports/confusion_light_path.png",
                "heavy_confusion": "reports/confusion_heavy_path.png",
                "latency_summary": "reports/latency_summary.png",
            }
        },
        "automation": {
            "command": f"python evaluate.py --base-path {args.base_path} --out-dir {args.out_dir} --strict-split-check --split-manifest {args.split_manifest}"
        },
    }

    if args.strict_split_check:
        split_report = _validate_split_manifest(
            os.path.join(base, args.split_manifest),
            os.path.join(base, "datasets"),
        )
        split_latest = os.path.join(logs_dir, "split_validation_report_latest.json")
        with open(split_latest, "w", encoding="utf-8") as f:
            json.dump(split_report, f, indent=2)

    out_latest = os.path.join(out_dir, "final_metrics_latest.json")
    out_plain = os.path.join(out_dir, "final_metrics.json")
    for path in (out_latest, out_plain):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(f"Saved: {out_latest}")
    print(f"Saved: {out_plain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
