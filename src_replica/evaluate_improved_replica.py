import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, matthews_corrcoef
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Adjust path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.data_resolvers import resolve_can_csv, resolve_eth_packet_csv
from src_replica.hybrid_curriculum import evaluation_pair_specs
from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD


def binary_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def binary_fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0


def binary_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def binary_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0


def bootstrap_ci(y_true, y_pred, n_resamples=1000, seed=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)

    f1_vals, mcc_vals, fpr_vals, fnr_vals = [], [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1_vals.append(f1_score(yt, yp, zero_division=0))
        mcc_vals.append(matthews_corrcoef(yt, yp) if len(np.unique(yt)) > 1 else 0.0)
        fpr_vals.append(binary_fpr(yt, yp))
        fnr_vals.append(binary_fnr(yt, yp))

    return {
        "f1_ci95": [float(np.percentile(f1_vals, 2.5)), float(np.percentile(f1_vals, 97.5))],
        "mcc_ci95": [float(np.percentile(mcc_vals, 2.5)), float(np.percentile(mcc_vals, 97.5))],
        "fpr_ci95": [float(np.percentile(fpr_vals, 2.5)), float(np.percentile(fpr_vals, 97.5))],
        "fnr_ci95": [float(np.percentile(fnr_vals, 2.5)), float(np.percentile(fnr_vals, 97.5))],
    }


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": float(binary_fpr(y_true, y_pred)),
        "fnr": float(binary_fnr(y_true, y_pred)),
        "specificity": float(binary_specificity(y_true, y_pred)),
        "npv": float(binary_npv(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def predict_dataset(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                xc = batch['can'].to(device)
                xe = batch['eth'].to(device)
                batch_labels = batch['label'].to(device)
            else:
                (xc, xe), batch_labels = batch
                xc = xc.to(device)
                xe = xe.to(device)
                batch_labels = batch_labels.to(device)

            logits = model(xc, xe)
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return np.asarray(labels), np.asarray(preds)

def evaluate_improved(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Improved Light Model on {device}")
    
    can_features = STANDARD_CAN_FEATURES_16
    input_dim = len(can_features)

    model = TinyHybridStudent(input_dim=input_dim, hidden_dim=64, num_classes=2).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    subset_bootstrap_resamples = max(200, min(args.bootstrap_resamples, 400))
    per_pair = {}
    per_group_buffers = {}
    overall_labels = []
    overall_preds = []
    loaded_specs = []

    for spec in evaluation_pair_specs():
        can_path = resolve_can_csv(args.data_dir, spec.can_file, prefer_raw=True)
        eth_npy_path = os.path.join(args.data_dir, spec.eth_npy_file)
        eth_csv_path = resolve_eth_packet_csv(args.data_dir, spec.eth_npy_file)

        if not (can_path and os.path.exists(can_path) and eth_csv_path):
            print(f"Skipping eval pair {spec.name}; missing source files.")
            continue

        print(f"Loading eval pair: {spec.name} [{spec.group}]")
        try:
            ds = CorrelatedHybridVehicleDataset(
                can_csv_path=can_path,
                eth_packet_csv_path=eth_csv_path,
                eth_npy_path=eth_npy_path,
                can_features=can_features,
                can_window_size=CAN_WINDOW_SIZE_STANDARD,
                eth_window_size=ETH_WINDOW_SIZE_STANDARD,
                can_max_rows=args.max_rows
            )
        except Exception as e:
            print(f"Error loading {spec.name}: {e}")
            continue

        if len(ds) == 0:
            print(f"Skipping eval pair {spec.name}; aligned dataset is empty.")
            continue

        labels, preds = predict_dataset(model, ds, device)
        pair_metrics = compute_metrics(labels, preds)
        pair_metrics.update(bootstrap_ci(labels, preds, n_resamples=subset_bootstrap_resamples, seed=args.seed))
        pair_metrics.update({
            "group": spec.group,
            "can_file": spec.can_file,
            "eth_npy_file": spec.eth_npy_file,
        })
        per_pair[spec.name] = pair_metrics
        loaded_specs.append({
            "name": spec.name,
            "group": spec.group,
            "can_file": spec.can_file,
            "eth_npy_file": spec.eth_npy_file,
            "samples": int(len(labels)),
        })
        overall_labels.extend(labels.tolist())
        overall_preds.extend(preds.tolist())

        if spec.group not in per_group_buffers:
            per_group_buffers[spec.group] = {"labels": [], "preds": []}
        per_group_buffers[spec.group]["labels"].extend(labels.tolist())
        per_group_buffers[spec.group]["preds"].extend(preds.tolist())

    if not overall_labels:
        print("No evaluation data found.")
        return

    metrics = compute_metrics(overall_labels, overall_preds)
    metrics.update(bootstrap_ci(overall_labels, overall_preds, n_resamples=args.bootstrap_resamples, seed=args.seed))
    metrics["per_pair"] = per_pair
    metrics["per_group"] = {}
    for group_name, buffers in per_group_buffers.items():
        group_metrics = compute_metrics(buffers["labels"], buffers["preds"])
        group_metrics.update(
            bootstrap_ci(
                buffers["labels"],
                buffers["preds"],
                n_resamples=subset_bootstrap_resamples,
                seed=args.seed,
            )
        )
        metrics["per_group"][group_name] = group_metrics
    metrics["curriculum_pairs"] = loaded_specs

    print("\n--- Improved Model Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    if len(set(overall_labels)) > 1:
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"MCC:       {metrics['mcc']:.4f}")
        print(f"FPR:       {metrics['fpr']:.4f}")
        print(f"FNR:       {metrics['fnr']:.4f}")

        cm = np.array(metrics['confusion_matrix'])
        print("Confusion Matrix:")
        print(cm)
        print("Per-group summary:")
        for group_name, group_metrics in metrics["per_group"].items():
            print(
                f"  {group_name}: "
                f"F1={group_metrics['f1']:.4f}, "
                f"Recall={group_metrics['recall']:.4f}, "
                f"FPR={group_metrics['fpr']:.4f}, "
                f"Samples={group_metrics['samples']}"
            )
    else:
        print("Single class in evaluation set - skipping F1/CM.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "improved_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation report to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--model_path", type=str, default="models/student_tiny_improved.pth")
    # Set default None to load full files
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    args = parser.parse_args()
    
    evaluate_improved(args)

if __name__ == "__main__":
    main()
