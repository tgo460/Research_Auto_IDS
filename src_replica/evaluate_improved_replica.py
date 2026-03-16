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

def evaluate_improved(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Improved Light Model on {device}")
    
    can_features = ['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
                    'can_id_freq_global', 'can_id_freq_win', 'payload_entropy', 
                    'inter_arrival', 'inter_arrival_roll_mean', 'id_switch_rate_win']
    input_dim = len(can_features)

    model = TinyHybridStudent(input_dim=input_dim, hidden_dim=64, num_classes=2).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Load Data (Engineered) - Validation Set (e.g. Fuzzy or Gear)
    # Using Fuzzy for validation
    engineered_dir = os.path.join(args.data_dir, "replica_can_b1_engineered")
    datasets_list = []
    
    pairs = [
        ("DoS", "can_dos_train.csv", "eth_driving_01_injected_images-003.npy", "eth_driving_01_injected.csv"),
        ("Fuzzy", "can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy", "eth_driving_02_injected.csv"),
        ("Gear", "can_gear_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
        ("RPM", "can_rpm_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
    ]
    
    loaded_files = set()
    
    per_attack_datasets = []

    for attack_name, can_f, eth_n, eth_c in pairs:
        if can_f in loaded_files: continue
        
        can_path = os.path.join(engineered_dir, can_f)
        eth_npy_path = os.path.join(args.data_dir, eth_n)
        
        # Robust ETH CSV search
        base_c = eth_c.replace(".csv", "")
        candidates = [
            os.path.join(args.data_dir, "replica_eth_smoke", f"{base_c}_replica_packets.csv"),
            os.path.join(args.data_dir, f"{base_c}_replica_packets.csv"),
            os.path.join(args.data_dir, f"{base_c}_preprocessed.csv"),
            os.path.join(args.data_dir, eth_c)
        ]
        
        eth_csv_path = None
        for cand in candidates:
            if os.path.exists(cand):
                eth_csv_path = cand
                break

        if os.path.exists(can_path) and os.path.exists(eth_npy_path) and eth_csv_path:
            print(f"Loading eval pair: {can_f}")
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
                if len(ds) > 0:
                    datasets_list.append(ds)
                    per_attack_datasets.append((attack_name, ds))
                    loaded_files.add(can_f)
            except Exception as e:
                print(f"Error: {e}")

    if not datasets_list:
        print("No evaluation data found.")
        return

    val_loader = DataLoader(ConcatDataset(datasets_list), batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                    xc = batch['can'].to(device)
                    xe = batch['eth'].to(device)
                    labels = batch['label'].to(device)
            else:
                    (xc, xe), labels = batch
                    xc = xc.to(device)
                    xe = xe.to(device)
                    labels = labels.to(device)
            
            logits = model(xc, xe)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics.update(bootstrap_ci(all_labels, all_preds, n_resamples=args.bootstrap_resamples, seed=args.seed))

    print("\n--- Improved Model Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    if len(set(all_labels)) > 1:
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"MCC:       {metrics['mcc']:.4f}")
        print(f"FPR:       {metrics['fpr']:.4f}")
        print(f"FNR:       {metrics['fnr']:.4f}")

        cm = np.array(metrics['confusion_matrix'])
        print("Confusion Matrix:")
        print(cm)

        # Per-attack metrics (publication-standard breakdown)
        per_attack = {}
        for attack_name, ds in per_attack_datasets:
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            a_preds, a_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, dict):
                        xc = batch['can'].to(device)
                        xe = batch['eth'].to(device)
                        labels = batch['label'].to(device)
                    else:
                        (xc, xe), labels = batch
                        xc = xc.to(device)
                        xe = xe.to(device)
                        labels = labels.to(device)
                    logits = model(xc, xe)
                    preds = torch.argmax(logits, dim=1)
                    a_preds.extend(preds.cpu().numpy())
                    a_labels.extend(labels.cpu().numpy())

            if len(a_labels) > 0 and len(np.unique(a_labels)) > 1:
                per_attack[attack_name] = compute_metrics(a_labels, a_preds)
            else:
                per_attack[attack_name] = {
                    "samples": int(len(a_labels)),
                    "note": "single class in this subset; binary metrics not fully defined",
                }

        metrics['per_attack'] = per_attack

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "improved_eval_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation report to {out_path}")
    else:
        print("Single class in evaluation set - skipping F1/CM.")

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
