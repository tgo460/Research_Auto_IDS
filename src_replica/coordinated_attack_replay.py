"""Coordinated attack replay -- multi-scenario test suite (P3).

Tests the full IDS cascade across multiple attack scenarios:
  * CAN-only attacks (DoS, Fuzzy, Gear, RPM) paired with normal ETH
  * ETH-only attacks (injected driving/indoors) paired with normal CAN
  * Coordinated attacks (attack CAN + attack ETH simultaneously)
  * Baseline (normal CAN + normal ETH)

Each scenario runs >=50 samples.  The report captures per-scenario
detection rate, FPR, and latency percentiles.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.runtime.adapters import CsvCanIngest, CsvEthIngest, FileAlertEgress, WatchdogHealthMonitor
from src_replica.runtime.config import load_deployment_config
from src_replica.runtime.engine import RuntimeIDSService

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
def _build_scenarios(can_dir: str, eth_dir: str) -> List[Dict]:
    """Return a list of scenario dicts with can_csv, eth_csv, name, expected_label."""
    attack_can = {
        "dos":   os.path.join(can_dir, "can_dos_train.csv"),
        "fuzzy": os.path.join(can_dir, "can_fuzzy_train.csv"),
        "gear":  os.path.join(can_dir, "can_gear_train.csv"),
        "rpm":   os.path.join(can_dir, "can_rpm_train.csv"),
    }
    normal_can = os.path.join(can_dir, "can_normal_train.csv")

    attack_eth = {
        "driving_01_inj": os.path.join(eth_dir, "eth_driving_01_injected_replica_packets.csv"),
        "driving_02_inj": os.path.join(eth_dir, "eth_driving_02_injected_replica_packets.csv"),
        "indoors_01_inj": os.path.join(eth_dir, "eth_indoors_01_injected_replica_packets.csv"),
        "indoors_02_inj": os.path.join(eth_dir, "eth_indoors_02_injected_replica_packets.csv"),
    }
    normal_eth = {
        "driving_01_orig": os.path.join(eth_dir, "eth_driving_01_original_replica_packets.csv"),
        "driving_02_orig": os.path.join(eth_dir, "eth_driving_02_original_replica_packets.csv"),
        "indoors_01_orig": os.path.join(eth_dir, "eth_indoors_01_original_replica_packets.csv"),
    }

    scenarios: List[Dict] = []

    # 1. Baseline (normal + normal) -- expected label 0
    for eth_name, eth_path in normal_eth.items():
        scenarios.append({
            "name": f"baseline_normal_can+{eth_name}",
            "can_csv": normal_can,
            "eth_csv": eth_path,
            "expected_label": 0,
        })

    # 2. CAN-only attacks (attack CAN + normal ETH) -- expected label 1
    for atk_name, can_path in attack_can.items():
        for eth_name, eth_path in normal_eth.items():
            scenarios.append({
                "name": f"can_only_{atk_name}+{eth_name}",
                "can_csv": can_path,
                "eth_csv": eth_path,
                "expected_label": 1,
            })

    # 3. ETH-only attacks (normal CAN + injected ETH) -- expected label 1
    for eth_name, eth_path in attack_eth.items():
        scenarios.append({
            "name": f"eth_only_normal_can+{eth_name}",
            "can_csv": normal_can,
            "eth_csv": eth_path,
            "expected_label": 1,
        })

    # 4. Coordinated attacks (attack CAN + injected ETH) -- expected label 1
    for atk_name, can_path in attack_can.items():
        for eth_name, eth_path in attack_eth.items():
            scenarios.append({
                "name": f"coordinated_{atk_name}+{eth_name}",
                "can_csv": can_path,
                "eth_csv": eth_path,
                "expected_label": 1,
            })

    return scenarios


@lru_cache(maxsize=None)
def _first_positive_can_row(can_csv: str) -> int:
    try:
        labels = pd.read_csv(can_csv, usecols=["Label"])["Label"]
    except Exception:
        return 0
    numeric = pd.to_numeric(labels, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    positive = np.flatnonzero(numeric > 0)
    return int(positive[0]) if positive.size > 0 else 0


def _scenario_offsets(scenario: Dict, can_window_size: int) -> Dict[str, int]:
    can_start_row = 0
    eth_start_row = 0
    name = scenario["name"]

    # CAN-only and coordinated scenarios should begin close to the first positive CAN
    # label so the replay budget is spent on the attack region rather than the long
    # benign prefix that exists in the raw attack traces.
    if name.startswith("can_only_") or name.startswith("coordinated_"):
        first_positive = _first_positive_can_row(scenario["can_csv"])
        if first_positive > 0:
            can_start_row = max(0, first_positive - int(can_window_size))

    return {
        "can_start_row": int(can_start_row),
        "eth_start_row": int(eth_start_row),
    }


def _run_scenario(cfg_path: str, scenario: Dict, max_samples: int) -> Dict:
    """Run a single scenario through the RuntimeIDSService and return metrics."""
    can_csv = scenario["can_csv"]
    eth_csv = scenario["eth_csv"]

    if not os.path.exists(can_csv) or not os.path.exists(eth_csv):
        return {
            "name": scenario["name"],
            "skipped": True,
            "reason": f"missing file: can={os.path.exists(can_csv)}, eth={os.path.exists(eth_csv)}",
        }

    cfg = load_deployment_config(cfg_path)
    cfg.can_source = can_csv
    cfg.eth_source = eth_csv
    cfg.max_samples = max_samples
    offsets = _scenario_offsets(scenario, cfg.can_window_size)

    try:
        svc = RuntimeIDSService(
            cfg=cfg,
            can_ingest=CsvCanIngest(cfg.can_source, start_row=offsets["can_start_row"]),
            eth_ingest=CsvEthIngest(cfg.eth_source, start_row=offsets["eth_start_row"]),
            alert_egress=FileAlertEgress(cfg.alert_output_path),
            health_monitor=WatchdogHealthMonitor(latency_budget_ms=cfg.latency_budget_ms),
        )
    except Exception as exc:
        return {
            "name": scenario["name"],
            "skipped": True,
            "reason": str(exc),
        }

    y_true, y_pred, lat = svc.run(max_samples=max_samples)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n_attack = int(np.sum(y_true == 1))
    n_normal = int(np.sum(y_true == 0))
    dr = float(np.mean(y_pred[y_true == 1] == 1)) if n_attack > 0 else None
    fpr = float(np.mean(y_pred[y_true == 0] == 1)) if n_normal > 0 else None
    fnr = float(np.mean(y_pred[y_true == 1] == 0)) if n_attack > 0 else None
    acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else None

    return {
        "name": scenario["name"],
        "expected_label": scenario["expected_label"],
        "samples": int(len(y_true)),
        "attack_samples": n_attack,
        "normal_samples": n_normal,
        "can_start_row": offsets["can_start_row"],
        "eth_start_row": offsets["eth_start_row"],
        "attack_not_reached": bool(scenario["expected_label"] == 1 and n_attack == 0),
        "detection_rate": dr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": acc,
        "latency_ms": {
            "mean": float(np.mean(lat)) if lat else 0.0,
            "p50": float(np.percentile(lat, 50)) if lat else 0.0,
            "p95": float(np.percentile(lat, 95)) if lat else 0.0,
            "max": float(np.max(lat)) if lat else 0.0,
        } if lat else {},
        "skipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-scenario coordinated attack replay (P3).")
    parser.add_argument("--config", default="configs/deployment.example.json")
    parser.add_argument("--can_dir", default="datasets",
                        help="Directory containing CAN replay CSVs.")
    parser.add_argument("--eth_dir", default=os.path.join("datasets", "replica_eth_smoke"),
                        help="Directory containing labeled ETH replay CSVs.")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Samples per scenario (large enough to reach attack regions)")
    parser.add_argument("--output", default="reports/coordinated_attack_report.json")
    args = parser.parse_args()

    scenarios = _build_scenarios(args.can_dir, args.eth_dir)
    print(f"=== Coordinated Attack Replay: {len(scenarios)} scenarios, {args.max_samples} samples each ===\n")

    results: List[Dict] = []
    total_attack_detected = 0
    total_attack_samples = 0
    total_normal_correct = 0
    total_normal_samples = 0

    for i, sc in enumerate(scenarios):
        print(f"[{i + 1}/{len(scenarios)}] {sc['name']} ... ", end="", flush=True)
        res = _run_scenario(args.config, sc, args.max_samples)
        results.append(res)
        if res.get("skipped"):
            print(f"SKIPPED ({res.get('reason', '')})")
            continue
        # Aggregate
        if res["attack_samples"] > 0 and res["detection_rate"] is not None:
            total_attack_detected += int(round(res["detection_rate"] * res["attack_samples"]))
            total_attack_samples += res["attack_samples"]
        if res["normal_samples"] > 0 and res["false_positive_rate"] is not None:
            total_normal_correct += int(round((1 - res["false_positive_rate"]) * res["normal_samples"]))
            total_normal_samples += res["normal_samples"]

        tag = f"DR={res['detection_rate']:.2%}" if res["detection_rate"] is not None else "DR=N/A"
        fpr_tag = f"FPR={res['false_positive_rate']:.2%}" if res["false_positive_rate"] is not None else "FPR=N/A"
        print(f"{res['samples']} samples | {tag} | {fpr_tag}")

    overall_dr = total_attack_detected / total_attack_samples if total_attack_samples > 0 else None
    overall_fpr = 1 - (total_normal_correct / total_normal_samples) if total_normal_samples > 0 else None

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type": "coordinated_attack_replay_multi_scenario",
        "num_scenarios": len(scenarios),
        "scenarios_run": sum(1 for r in results if not r.get("skipped")),
        "scenarios_skipped": sum(1 for r in results if r.get("skipped")),
        "attack_scenarios_without_positive_labels": sum(
            1 for r in results if r.get("expected_label") == 1 and r.get("attack_not_reached")
        ),
        "overall_detection_rate": overall_dr,
        "overall_false_positive_rate": overall_fpr,
        "total_attack_samples": total_attack_samples,
        "total_normal_samples": total_normal_samples,
        "per_scenario": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Overall DR:  {overall_dr:.2%}" if overall_dr is not None else "Overall DR:  N/A")
    print(f"Overall FPR: {overall_fpr:.2%}" if overall_fpr is not None else "Overall FPR: N/A")
    print(f"Report saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
