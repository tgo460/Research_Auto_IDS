import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.architecture_improved import TinyHybridStudent
from src_replica.cascade_eval_replica import build_mixed_dataset
from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16


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


class ModalityAblationDataset(Dataset):
    def __init__(self, base_dataset: Dataset, mode: str):
        allowed = {"fused", "can_only", "eth_only"}
        if mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}, got {mode!r}")
        self.base_dataset = base_dataset
        self.mode = mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError("ModalityAblationDataset expects dict-style samples")

        can = sample["can"].clone()
        eth = sample["eth"].clone()
        if self.mode == "can_only":
            eth = torch.zeros_like(eth)
        elif self.mode == "eth_only":
            can = torch.zeros_like(can)

        out = dict(sample)
        out["can"] = can
        out["eth"] = eth
        return out


def _predict_dataset(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            xc = batch["can"].to(device)
            xe = batch["eth"].to(device)
            y = batch["label"].to(device)
            logits = model(xc, xe)
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.asarray(labels), np.asarray(preds)


def evaluate_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyHybridStudent(input_dim=len(STANDARD_CAN_FEATURES_16), hidden_dim=64, num_classes=2).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    report: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "strict_ablation_eval",
        "model_path": args.model_path,
        "split_manifest": args.split_manifest,
        "pairing_mode": args.pairing_mode,
        "splits": {},
    }

    for split in args.splits:
        ds = build_mixed_dataset(
            base_path=ROOT_DIR,
            split=split,
            split_manifest=args.split_manifest,
            pairing_mode=args.pairing_mode,
            require_both_classes=True,
        )
        split_row: Dict[str, object] = {}
        if ds is None or len(ds) == 0:
            split_row["error"] = "empty dataset"
            report["splits"][split] = split_row
            continue

        metadata = getattr(ds, "metadata", {})
        split_row["dataset"] = {
            "samples": int(len(ds)),
            "metadata": metadata,
        }
        split_row["modes"] = {}

        for mode in args.modes:
            mode_ds = ModalityAblationDataset(ds, mode=mode)
            labels, preds = _predict_dataset(model, mode_ds, device=device, batch_size=args.batch_size)
            metrics = _metrics(labels, preds)
            metrics.update(_bootstrap_ci(labels, preds, n_resamples=args.bootstrap_resamples, seed=args.seed))
            split_row["modes"][mode] = metrics

        report["splits"][split] = split_row

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "strict_ablation_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {out_path}")


def _csv_arg(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser(description="Strict split ablation evaluation for fused, CAN-only, and ETH-only modes.")
    parser.add_argument("--model_path", type=str, default="models/student_tiny_improved.pth")
    parser.add_argument("--split_manifest", type=str, default=os.path.join("data", "splits", "split_v3_research_valid.json"))
    parser.add_argument("--pairing_mode", type=str, default="label_cartesian", choices=["label_cartesian", "single_match"])
    parser.add_argument("--splits", type=_csv_arg, default=["val", "test"])
    parser.add_argument("--modes", type=_csv_arg, default=["fused", "can_only", "eth_only"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    evaluate_ablation(args)


if __name__ == "__main__":
    main()
