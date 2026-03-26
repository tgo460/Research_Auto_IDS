import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.cascade_eval_replica import build_mixed_dataset


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


def _extract_features_and_labels(dataset, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode not in {"can_only", "eth_only"}:
        raise ValueError(f"mode must be 'can_only' or 'eth_only', got {mode!r}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError("Expected dict-style samples from build_mixed_dataset")
        tensor = sample["can"] if mode == "can_only" else sample["eth"]
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        features.append(arr)
        labels.append(int(sample["label"]))

    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X = np.stack(features).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def _make_estimator(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _csv_arg(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _load_split_dataset(split: str, split_manifest: str, pairing_mode: str):
    return build_mixed_dataset(
        base_path=ROOT_DIR,
        split=split,
        split_manifest=split_manifest,
        pairing_mode=pairing_mode,
        require_both_classes=True,
    )


def evaluate_baselines(args):
    report: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "unimodal_baseline_eval",
        "split_manifest": args.split_manifest,
        "pairing_mode": args.pairing_mode,
        "modes": {},
    }

    train_ds = _load_split_dataset("train", args.split_manifest, args.pairing_mode)
    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("Strict train split is empty; cannot train unimodal baselines.")

    train_meta = getattr(train_ds, "metadata", {})

    for mode in args.modes:
        X_train, y_train = _extract_features_and_labels(train_ds, mode)
        if X_train.size == 0:
            report["modes"][mode] = {"error": "empty train dataset"}
            continue

        model = _make_estimator(seed=args.seed)
        model.fit(X_train, y_train)

        mode_report: Dict[str, object] = {
            "train": {
                "dataset": {
                    "samples": int(len(train_ds)),
                    "metadata": train_meta,
                    "feature_dim": int(X_train.shape[1]),
                }
            },
            "eval": {},
        }

        train_pred = model.predict(X_train)
        train_metrics = _metrics(y_train, train_pred)
        train_metrics.update(_bootstrap_ci(y_train, train_pred, n_resamples=min(args.bootstrap_resamples, 400), seed=args.seed))
        mode_report["train"]["metrics"] = train_metrics

        for split in args.eval_splits:
            ds = _load_split_dataset(split, args.split_manifest, args.pairing_mode)
            if ds is None or len(ds) == 0:
                mode_report["eval"][split] = {"error": "empty dataset"}
                continue

            X_eval, y_eval = _extract_features_and_labels(ds, mode)
            pred = model.predict(X_eval)
            metrics = _metrics(y_eval, pred)
            metrics.update(_bootstrap_ci(y_eval, pred, n_resamples=args.bootstrap_resamples, seed=args.seed))
            mode_report["eval"][split] = {
                "dataset": {
                    "samples": int(len(ds)),
                    "metadata": getattr(ds, "metadata", {}),
                    "feature_dim": int(X_eval.shape[1]),
                },
                "metrics": metrics,
            }

        report["modes"][mode] = mode_report

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "unimodal_baseline_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate strict CAN-only and ETH-only baselines on the same split manifest.")
    parser.add_argument("--split_manifest", type=str, default=os.path.join("data", "splits", "split_v3_research_valid.json"))
    parser.add_argument("--pairing_mode", type=str, default="label_cartesian", choices=["label_cartesian", "single_match"])
    parser.add_argument("--modes", type=_csv_arg, default=["can_only", "eth_only"])
    parser.add_argument("--eval_splits", type=_csv_arg, default=["val", "test"])
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    evaluate_baselines(args)


if __name__ == "__main__":
    main()
