import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD
from src_replica.loao_train_replica import CAN_FEATURES_16, _load_pair

def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif"
    })

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_confidences(model, dataloader, device):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            out = model(b_can, b_eth)
            
            # Use softmax probability for class 1 (attack)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to a trained model")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--out_dir", type=str, default="reports/paper_figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=2000)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming the provided model was trained with 'dos' and 'fuzzy', and 'gear' is the holdout/unknown
    attacks = {
        "dos": {"can": "can_dos_train.csv", "eth": "eth_driving_01_injected_preprocessed.csv", "img": "eth_driving_01_injected_images-003.npy"},
        "fuzzy": {"can": "can_fuzzy_train.csv", "eth": "eth_driving_02_injected_preprocessed.csv", "img": "eth_driving_02_injected_images-008.npy"},
    }
    
    unknown_attack = {
        "can": "can_gear_train.csv", "eth": "eth_indoors_01_injected_preprocessed.csv", "img": "eth_indoors_01_injected_images.npy"
    }

    normal = {
        "can": "can_normal_train.csv",
        "eth": "eth_driving_01_original_preprocessed.csv",
        "img": "eth_driving_01_original_images-006.npy"
    }

    print("Loading datasets...")
    # Load normal
    n_ds = _load_pair(args.data_dir, normal["can"], normal["img"], max_rows=args.max_rows)
    # Load known attacks
    k_ds_list = []
    for k, v in attacks.items():
        ds = _load_pair(args.data_dir, v["can"], v["img"], max_rows=args.max_rows)
        if ds is not None:
             k_ds_list.append(ds)
    k_ds = ConcatDataset(k_ds_list) if k_ds_list else None
    
    # Load unknown attack
    u_ds = _load_pair(args.data_dir, unknown_attack["can"], unknown_attack["img"], max_rows=args.max_rows)

    model = TinyHybridStudent(
        input_dim=len(CAN_FEATURES_16),
        hidden_dim=32,
        num_classes=2
    )

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model path {args.model_path} not found. Exiting.")
        return
        
    model.to(device)
    
    print("Computing confidences...")
    confs_n = get_confidences(model, DataLoader(n_ds, batch_size=64), device) if n_ds else np.array([])
    confs_k = get_confidences(model, DataLoader(k_ds, batch_size=64), device) if k_ds else np.array([])
    confs_u = get_confidences(model, DataLoader(u_ds, batch_size=64), device) if u_ds else np.array([])

    set_style()
    plt.figure(figsize=(10, 6))

    if len(confs_n) > 0:
        sns.kdeplot(confs_n, fill=True, label="Normal Traffic", color="green", alpha=0.3)
    if len(confs_k) > 0:
        sns.kdeplot(confs_k, fill=True, label="Known Attacks (Training Set)", color="blue", alpha=0.3)
    if len(confs_u) > 0:
        sns.kdeplot(confs_u, fill=True, label="Unknown Attack (Out-of-Distribution)", color="red", alpha=0.3)

    # Formal Research OOD Metrics (Zero-day vs Normal)
    fpr95 = 0.0
    if len(confs_n) > 0 and len(confs_u) > 0:
        y_true = np.concatenate([np.zeros(len(confs_n)), np.ones(len(confs_u))])
        y_scores = np.concatenate([confs_n, confs_u])
        
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate FPR at 95% TPR
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            fpr95 = fpr[idx[0]]
            threshold95 = thresholds[idx[0]]
            
            # Plot dynamic tuned threshold instead of hardcoded 0.5
            plt.axvline(x=threshold95, color='orange', linestyle='--', label=f'OOD Threshold (95% TPR): {threshold95:.2f}')
            
            # Print standard metrics
            print("\n--- OOD Detection Performance (Normal vs Unknown) ---")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            print(f"FPR@95%TPR: {fpr95:.4f} (Threshold: {threshold95:.4f})")
            print("---------------------------------------------------\n")

    # Plot baseline decision threshold
    plt.axvline(x=0.5, color='k', linestyle=':', label='Baseline Threshold (0.5)')

    plt.title(f"Model Confidence Distribution (OOD Analysis)\nAUROC: {auroc:.3f} | AUPRC: {auprc:.3f} | FPR@95%TPR: {fpr95:.3f}" if 'auroc' in locals() else "Model Confidence Distribution (OOD Analysis)")
    plt.xlabel("Predicted Probability of Attack (Class 1)")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()

    out_path = os.path.join(args.out_dir, "ood_confidence_distribution_replica.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved OOD visualization to {out_path}")

if __name__ == "__main__":
    main()