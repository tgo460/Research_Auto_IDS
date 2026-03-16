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
from sklearn.metrics import accuracy_score, f1_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
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

def apply_feature_noise(can_tensor, epsilon, noise_indices):
    """
    Simulates a black-box adversarial timing/entropy evasion attack
    by injecting bounded Gaussian noise dynamically scaled to the feature's standard deviation.
    """
    noisy_tensor = can_tensor.clone()
    for idx in noise_indices:
        # Calculate standard deviation for proportional scaling
        std = noisy_tensor[:, :, idx].std(dim=(0, 1), keepdim=True)
        # Avoid zero-std collapsing
        std = torch.clamp(std, min=1e-5)
        
        noise = epsilon * std * torch.randn_like(noisy_tensor[:, :, idx])
        noisy_tensor[:, :, idx] += noise
    return noisy_tensor

def evaluate_robustness(model, dataloader, device, epsilon, continuous_indices):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            b_y = batch["label"].to(device)
            
            if epsilon > 0:
                b_can = apply_feature_noise(b_can, epsilon, continuous_indices)
            
            out = model(b_can, b_eth)
            preds = torch.argmax(out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_y.cpu().numpy())
            
    yl = np.array(all_labels)
    yp = np.array(all_preds)
    
    # We focus primarily on Recall (Detection Rate) to measure evasion success
    acc = accuracy_score(yl, yp)
    f1 = f1_score(yl, yp, zero_division=0)
    rec = recall_score(yl, yp, zero_division=0)
    
    return {"accuracy": float(acc), "f1": float(f1), "recall": float(rec)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained FP32 model")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--max_rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "paper_figures"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Known continuous/statistical features in CAN_FEATURES_16 that an attacker might tweak:
    # 10: can_id_freq_global, 11: can_id_freq_win, 12: payload_entropy, 
    # 13: inter_arrival, 14: inter_arrival_roll_mean, 15: id_switch_rate_win
    continuous_indices = [10, 11, 12, 13, 14, 15]

    print(f"Loading Base Dataset for Adversarial Evaluation (Holding Max Rows: {args.max_rows})...")
    # Load a mix of Normal and Attack Data 
    dataset = _load_pair(
        args.data_dir, 
        "can_dos_train.csv", 
        "eth_driving_01_injected_images-003.npy", 
        max_rows=args.max_rows
    )
    if dataset is None:
        print("Required dataset missing!")
        return

    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Load Model
    model = TinyHybridStudent(
        input_dim=len(CAN_FEATURES_16),
        hidden_dim=32,
        num_classes=2
    ).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded Model: {args.model_path}")
    else:
        print(f"Model path {args.model_path} not found.")
        return

    # Epsilons (Noise intensity multiplier)
    epsilons = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    results = {}
    
    print("\nStarting Adversarial Perturbation Tests...")
    for eps in epsilons:
        metrics = evaluate_robustness(model, loader, device, eps, continuous_indices)
        results[eps] = metrics
        print(f"  Epsilon {eps:.2f} -> Recall (Detection Rate): {metrics['recall']*100:.1f}%, F1: {metrics['f1']:.3f}")
        
    # Plotting Deployment
    set_style()
    plt.figure(figsize=(8, 5))
    
    eps_keys = list(results.keys())
    rec_vals = [results[e]["recall"] for e in eps_keys]
    f1_vals = [results[e]["f1"] for e in eps_keys]

    plt.plot(eps_keys, rec_vals, 's-', color="#d62728", label="Recall (Detection Rate)", linewidth=2.5, markersize=8)
    plt.plot(eps_keys, f1_vals, 'o--', color="#1f77b4", label="F1 Score", linewidth=2.0)
    
    plt.axhline(y=0.5, color='gray', linestyle=':', label='50% Threshold')
    
    plt.xlabel(r"Noise Multiplier ($\epsilon$) on Timing/Statistical Features", fontweight='bold')
    plt.ylabel("Performance Metric", fontweight='bold')
    plt.title("Model Robustness Against Timing-based Evasion Attacks", fontsize=15, pad=15)
    plt.ylim(0, 1.05)
    plt.xlim(0, max(eps_keys))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    plot_path = os.path.join(args.out_dir, "paper_figures", "adversarial_evasion_robustness_replica.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Save Report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_tested": args.model_path,
        "dataset": "dos_mix",
        "perturbed_indices": continuous_indices,
        "results": {str(k): v for k, v in results.items()}
    }
    
    report_path = os.path.join(args.out_dir, "adversarial_robustness_report_replica.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nSaved Robustness Figure to: {plot_path}")
    print(f"Saved Report JSON to: {report_path}")

if __name__ == "__main__":
    main()