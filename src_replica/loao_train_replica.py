import argparse
import json
import os
import sys
import numpy as np
import copy
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

CAN_FEATURES_16 = [
    "CAN_ID", "DLC", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "can_id_freq_global", "can_id_freq_win", "payload_entropy",
    "inter_arrival", "inter_arrival_roll_mean", "id_switch_rate_win",
]

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _eth_csv_from_npy(data_dir: str, eth_npy_file: str):
    base = eth_npy_file.split("_images")[0] if "_images" in eth_npy_file else os.path.splitext(eth_npy_file)[0]
    candidates = [
        os.path.join(data_dir, "replica_eth_smoke", f"{base}_replica_packets.csv"),
        os.path.join(data_dir, f"{base}_replica_packets.csv"),
        os.path.join(data_dir, f"{base}_preprocessed.csv"),
        os.path.join(data_dir, f"{base}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _load_pair(data_dir: str, can_file: str, eth_npy_file: str, max_rows=None):
    can_csv = os.path.join(data_dir, "replica_can_b1_engineered", can_file)
    if not os.path.exists(can_csv):
        can_csv = os.path.join(data_dir, can_file)
    eth_npy = os.path.join(data_dir, eth_npy_file)
    eth_csv = _eth_csv_from_npy(data_dir, eth_npy_file)

    if not (os.path.exists(can_csv) and os.path.exists(eth_npy) and eth_csv and os.path.exists(eth_csv)):
        return None

    return CorrelatedHybridVehicleDataset(
        can_csv_path=can_csv,
        eth_packet_csv_path=eth_csv,
        eth_npy_path=eth_npy,
        can_features=CAN_FEATURES_16,
        can_window_size=CAN_WINDOW_SIZE_STANDARD,
        eth_window_size=ETH_WINDOW_SIZE_STANDARD,
        can_max_rows=max_rows
    )

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            b_y = batch["label"].to(device)
            out = model(b_can, b_eth)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_y.cpu().numpy())
            
    yl = np.array(all_labels)
    yp = np.array(all_preds)
    
    if len(yl) == 0:
        return {}
        
    acc = accuracy_score(yl, yp)
    f1 = f1_score(yl, yp, zero_division=0)
    prec = precision_score(yl, yp, zero_division=0)
    rec = recall_score(yl, yp, zero_division=0)
    mcc = matthews_corrcoef(yl, yp)
    cm = confusion_matrix(yl, yp, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "mcc": mcc,
        "fnr": fnr,
        "support_total": int(len(yl)),
        "support_attack": int(tp + fn)
    }

def train_loao_model(model, train_loader, val_loader, device, epochs, learning_rate):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            b_y = batch["label"].to(device).long()
            
            optimizer.zero_grad()
            out = model(b_can, b_eth)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        metrics = evaluate_model(model, val_loader, device)
        f1 = metrics.get('f1', 0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_rows", type=int, default=5000)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Define attack domains
    attacks = {
        "dos": {"can": "can_dos_train.csv", "eth": "eth_driving_01_injected_preprocessed.csv", "img": "eth_driving_01_injected_images-003.npy"},
        "fuzzy": {"can": "can_fuzzy_train.csv", "eth": "eth_driving_02_injected_preprocessed.csv", "img": "eth_driving_02_injected_images-008.npy"},
        "gear": {"can": "can_gear_train.csv", "eth": "eth_indoors_01_injected_preprocessed.csv", "img": "eth_indoors_01_injected_images.npy"}
    }
    
    normal = {
        "can": "can_normal_train.csv",
        "eth": "eth_driving_01_original_preprocessed.csv",
        "img": "eth_driving_01_original_images-006.npy"
    }
    
    results = {}
    
    for holdout_name, holdout_files in attacks.items():
        print(f"\\n--- LOAO Retraining: Holding out '{holdout_name}' ---")
        
        # Build training datasets
        train_datasets = []
        for name, files in attacks.items():
            if name == holdout_name:
                continue
            
            ds = _load_pair(args.data_dir, files["can"], files["img"], max_rows=args.max_rows)
            if ds is not None:
                train_datasets.append(ds)
        
        # Add normal to train
        n_ds = _load_pair(args.data_dir, normal["can"], normal["img"], max_rows=args.max_rows)
        if n_ds is not None:
            train_datasets.append(n_ds)
            
        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=True)
        
        # Build validation dataset (the held-out attack + subset of normal)
        val_datasets = []
        val_ds = _load_pair(args.data_dir, holdout_files["can"], holdout_files["img"], max_rows=args.max_rows)
        
        if val_ds is not None:
            val_datasets.append(val_ds)
            if n_ds is not None:
                val_datasets.append(n_ds)
        
        if not val_datasets:
            print(f"Skipping {holdout_name}, files not found.")
            continue
            
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False)
        
        # Initialize and Train
        model = TinyHybridStudent(
            input_dim=len(CAN_FEATURES_16),
            hidden_dim=32,
            num_classes=2
        ).to(device)
        
        print(f"Training on {len(attacks)-1} attacks + normal. Evaluating on {holdout_name} + normal...")
        model = train_loao_model(model, train_loader, val_loader, device, args.epochs, args.lr)
        
        # Final Evaluation
        metrics = evaluate_model(model, val_loader, device)
        results[holdout_name] = metrics
        
        print(f"Held out '{holdout_name}' zero-day performance:")
        print(f"  F1: {metrics.get('f1', 0):.4f}")
        print(f"  MCC: {metrics.get('mcc', 0):.4f}")
        print(f"  FNR: {metrics.get('fnr', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_save_path = os.path.join("models", f"loao_model_heldout_{holdout_name}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")
        
    # Save Report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "loao_results": results
    }
    
    out_path = os.path.join(args.output_dir, "loao_evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\\nReport saved to: {out_path}")

if __name__ == "__main__":
    main()