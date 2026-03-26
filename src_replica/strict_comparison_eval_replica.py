import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.ablation_eval_replica import evaluate_ablation
from src_replica.unimodal_baseline_eval_replica import evaluate_baselines


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_comparison(ablation_report: Dict, baseline_report: Dict) -> Dict:
    split_names = sorted(set(ablation_report.get("splits", {}).keys()))
    comparison = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "strict_comparison_eval",
        "split_manifest": ablation_report.get("split_manifest") or baseline_report.get("split_manifest"),
        "pairing_mode": ablation_report.get("pairing_mode") or baseline_report.get("pairing_mode"),
        "splits": {},
    }

    for split in split_names:
        ablation_split = ablation_report.get("splits", {}).get(split, {})
        modes = ablation_split.get("modes", {})
        split_row = {
            "hybrid_fused": modes.get("fused"),
            "hybrid_can_masked": modes.get("can_only"),
            "hybrid_eth_masked": modes.get("eth_only"),
            "baseline_can_only": baseline_report.get("modes", {}).get("can_only", {}).get("eval", {}).get(split, {}).get("metrics"),
            "baseline_eth_only": baseline_report.get("modes", {}).get("eth_only", {}).get("eval", {}).get(split, {}).get("metrics"),
        }
        comparison["splits"][split] = split_row

    return comparison


def evaluate_comparison(args):
    ablation_args = argparse.Namespace(
        model_path=args.hybrid_model_path,
        split_manifest=args.split_manifest,
        pairing_mode=args.pairing_mode,
        splits=args.eval_splits,
        modes=["fused", "can_only", "eth_only"],
        batch_size=args.batch_size,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    evaluate_ablation(ablation_args)

    baseline_args = argparse.Namespace(
        split_manifest=args.split_manifest,
        pairing_mode=args.pairing_mode,
        modes=["can_only", "eth_only"],
        eval_splits=args.eval_splits,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    evaluate_baselines(baseline_args)

    ablation_path = os.path.join(args.output_dir, "strict_ablation_eval_report.json")
    baseline_path = os.path.join(args.output_dir, "unimodal_baseline_eval_report.json")
    comparison = _build_comparison(_load_json(ablation_path), _load_json(baseline_path))

    out_path = os.path.join(args.output_dir, "strict_comparison_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved report to {out_path}")


def _csv_arg(value: str):
    return [token.strip() for token in value.split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run strict hybrid ablations and unimodal baselines, then merge them into one comparison report.")
    parser.add_argument("--hybrid_model_path", type=str, default="models/student_tiny_improved.pth")
    parser.add_argument("--split_manifest", type=str, default=os.path.join("data", "splits", "split_v3_research_valid.json"))
    parser.add_argument("--pairing_mode", type=str, default="label_cartesian", choices=["label_cartesian", "single_match"])
    parser.add_argument("--eval_splits", type=_csv_arg, default=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    evaluate_comparison(args)


if __name__ == "__main__":
    main()
