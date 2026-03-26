import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from src_replica.data_resolvers import resolve_can_csv, resolve_eth_packet_csv
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.hybrid_curriculum import evaluation_pair_specs
from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

CAN_FEATURES_16 = STANDARD_CAN_FEATURES_16


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_pair(data_dir: str, can_file: str, eth_npy_file: str, max_rows=None):
    can_csv = resolve_can_csv(data_dir, can_file, prefer_raw=True)
    eth_npy = os.path.join(data_dir, eth_npy_file)
    eth_csv = resolve_eth_packet_csv(data_dir, eth_npy_file)

    if not (can_csv and os.path.exists(can_csv) and eth_csv and os.path.exists(eth_csv)):
        return None

    return CorrelatedHybridVehicleDataset(
        can_csv_path=can_csv,
        eth_packet_csv_path=eth_csv,
        eth_npy_path=eth_npy if os.path.exists(eth_npy) else None,
        can_features=CAN_FEATURES_16,
        can_window_size=CAN_WINDOW_SIZE_STANDARD,
        eth_window_size=ETH_WINDOW_SIZE_STANDARD,
        can_max_rows=max_rows,
    )


def _binary_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def _binary_fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": float(_binary_fpr(y_true, y_pred)),
        "fnr": float(_binary_fnr(y_true, y_pred)),
        "specificity": float(specificity),
        "npv": float(npv),
    }


def _bootstrap_ci(y_true, y_pred, n_resamples=1000, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_vals, mcc_vals, fpr_vals, fnr_vals = [], [], [], []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1_vals.append(f1_score(yt, yp, zero_division=0))
        mcc_vals.append(matthews_corrcoef(yt, yp) if len(np.unique(yt)) > 1 else 0.0)
        fpr_vals.append(_binary_fpr(yt, yp))
        fnr_vals.append(_binary_fnr(yt, yp))

    return {
        "f1_ci95": [float(np.percentile(f1_vals, 2.5)), float(np.percentile(f1_vals, 97.5))],
        "mcc_ci95": [float(np.percentile(mcc_vals, 2.5)), float(np.percentile(mcc_vals, 97.5))],
        "fpr_ci95": [float(np.percentile(fpr_vals, 2.5)), float(np.percentile(fpr_vals, 97.5))],
        "fnr_ci95": [float(np.percentile(fnr_vals, 2.5)), float(np.percentile(fnr_vals, 97.5))],
    }


def _eval_dataset(model, dataset, device, bootstrap_resamples=1000, seed=42):
    if dataset is None or len(dataset) == 0:
        return {"error": "empty dataset"}

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    preds, labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                xc = batch["can"].to(device)
                xe = batch["eth"].to(device)
                y = batch["label"].to(device)
            else:
                (xc, xe), y = batch
                xc = xc.to(device)
                xe = xe.to(device)
                y = y.to(device)

            logits = model(xc, xe)
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    metrics = _metrics(labels, preds)
    metrics.update(_bootstrap_ci(labels, preds, n_resamples=bootstrap_resamples, seed=seed))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-domain and attack-holdout generalization evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyHybridStudent(input_dim=16, hidden_dim=64, num_classes=2).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Cross-domain protocol: same CAN classes, different ETH domain.
    domain_proto = {
        "driving": {
            "attack": ("can_dos_train.csv", "eth_driving_01_injected_images-003.npy"),
            "normal": ("can_normal_train.csv", "eth_driving_01_original_images-006.npy"),
        },
        "indoors": {
            "attack": ("can_dos_train.csv", "eth_indoors_01_injected_images.npy"),
            "normal": ("can_normal_train.csv", "eth_indoors_01_original_images.npy"),
        },
    }

    cross_domain = {}
    for domain_name, proto in domain_proto.items():
        ds_parts = []
        for _, (can_file, eth_file) in proto.items():
            ds = _load_pair(args.data_dir, can_file, eth_file, max_rows=args.max_rows)
            if ds is not None and len(ds) > 0:
                ds_parts.append(ds)
        domain_ds = ConcatDataset(ds_parts) if ds_parts else None
        cross_domain[domain_name] = _eval_dataset(
            model,
            domain_ds,
            device,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

    # Attack-holdout style reporting: evaluate each attack vs same normal baseline.
    holdout_proto = {
        "DoS": "can_dos_train.csv",
        "Fuzzy": "can_fuzzy_train.csv",
        "Gear": "can_gear_train.csv",
        "RPM": "can_rpm_train.csv",
    }

    holdout_eval = {}
    normal_ds = _load_pair(
        args.data_dir,
        "can_normal_train.csv",
        "eth_driving_01_original_images-006.npy",
        max_rows=args.max_rows,
    )

    for attack_name, can_attack in holdout_proto.items():
        attack_ds = _load_pair(
            args.data_dir,
            can_attack,
            "eth_driving_01_injected_images-003.npy",
            max_rows=args.max_rows,
        )
        ds_parts = []
        if normal_ds is not None and len(normal_ds) > 0:
            ds_parts.append(normal_ds)
        if attack_ds is not None and len(attack_ds) > 0:
            ds_parts.append(attack_ds)

        eval_ds = ConcatDataset(ds_parts) if ds_parts else None
        holdout_eval[attack_name] = _eval_dataset(
            model,
            eval_ds,
            device,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

    curriculum_group_parts = {}
    curriculum_pair_results = {}
    for spec in evaluation_pair_specs():
        ds = _load_pair(args.data_dir, spec.can_file, spec.eth_npy_file, max_rows=args.max_rows)
        curriculum_pair_results[spec.name] = {
            "group": spec.group,
            "can_file": spec.can_file,
            "eth_npy_file": spec.eth_npy_file,
        }
        if ds is None or len(ds) == 0:
            curriculum_pair_results[spec.name]["error"] = "empty dataset"
            continue

        curriculum_pair_results[spec.name].update(
            _eval_dataset(
                model,
                ds,
                device,
                bootstrap_resamples=min(args.bootstrap_resamples, 300),
                seed=args.seed,
            )
        )
        curriculum_group_parts.setdefault(spec.group, []).append(ds)

    curriculum_group_results = {}
    for group_name, parts in curriculum_group_parts.items():
        group_ds = ConcatDataset(parts) if parts else None
        curriculum_group_results[group_name] = _eval_dataset(
            model,
            group_ds,
            device,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "generalization_eval",
        "model_path": args.model_path,
        "protocol": {
            "cross_domain": "evaluate driving vs indoors domains with matched CAN attack/normal pairing",
            "attack_holdout": "evaluate each CAN attack against common normal baseline",
            "note": "attack_holdout here is evaluation protocol; true LOAO requires retraining with held-out attack excluded",
        },
        "cross_domain_results": cross_domain,
        "attack_holdout_results": holdout_eval,
        "curriculum_group_results": curriculum_group_results,
        "curriculum_pair_results": curriculum_pair_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "generalization_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
