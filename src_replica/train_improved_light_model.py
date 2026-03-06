import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score

# Adjust path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from architecture_improved import TinyHybridStudent
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

def train_light_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Improved Light Model on {device}")
    datasets_list = []
    
    # Define Features
    # Based on the engineered datasets
    can_features = ['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
                    'can_id_freq_global', 'can_id_freq_win', 'payload_entropy', 
                    'inter_arrival', 'inter_arrival_roll_mean', 'id_switch_rate_win']
    input_dim = len(can_features)
    print(f"Using {input_dim} CAN features.")

    # Model
    model = TinyHybridStudent(input_dim=input_dim, hidden_dim=64, num_classes=2).to(device)
    
    # Optimizer & Loss
    # Compute class weights from training data to handle imbalanced attack/normal ratios
    # (PDF standard: Use SMOTE or class weighting)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Load Data (Engineered)
    # We need to construct datasets manually since build_mixed_dataset in cascade_eval_replica 
    # points to raw data or is complex.
    
    # Data Mapping (Engineered CAN + NPY Images)
    # We will look for pairs in the engineered folder
    engineered_dir = os.path.join(args.data_dir, "replica_can_b1_engineered")
    
    # Map of CAN file -> ETH NPY file (heuristic)
    # can_dos_train.csv -> eth_driving_01_injected_images-003.npy (Dos)
    # can_normal_train.csv -> eth_driving_01_original_images-006.npy (Normal)
    # can_fuzzy_train.csv -> eth_driving_02_injected_images-008.npy (Fuzzy)
    
    pairs = [
        # (Engineered CAN, ETH NPY, ETH CSV Base)
        ("can_dos_train.csv", "eth_driving_01_injected_images-003.npy", "eth_driving_01_injected.csv"), # Attack 1
        ("can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy", "eth_driving_02_injected.csv"), # Attack 2
        ("can_gear_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"), # Attack 3 (Spoofing on Normal Eth?)
        ("can_rpm_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"), # Attack 4
        # Add Normal if possible? 
        # Since 'can_normal_train.csv' is missing from engineered folder, we rely on the normal segments within the attack files?
        # Or we load the raw 'can_normal_train.csv' and compute features on fly? Too complex for now.
    ]
    
    # Remove duplicates if any
    loaded_files = set()
    
    for can_f, eth_n, eth_c in pairs:
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
            print(f"Loading pair: {can_f} + {eth_n}")
            try:
                ds = CorrelatedHybridVehicleDataset(
                    can_csv_path=can_path,
                    eth_packet_csv_path=eth_csv_path,
                    eth_npy_path=eth_npy_path,
                    can_features=can_features,
                    can_window_size=CAN_WINDOW_SIZE_STANDARD,
                    eth_window_size=ETH_WINDOW_SIZE_STANDARD,
                    can_max_rows=args.max_rows # Will use full file if None
                )
                if len(ds) > 0:
                    datasets_list.append(ds)
                    loaded_files.add(can_f)
                    print(f"  -> Added {len(ds)} samples.")
                else:
                    print(f"  -> Warning: Dataset empty after alignment.")
            except Exception as e:
                print(f"Error loading {can_f}: {e}")
        else:
            print(f"Skipping pair {can_f}, missing files (ETH CSV found? {eth_csv_path is not None}).")


    if not datasets_list:
        print("No datasets loaded. Exiting.")
        return

    print("Computing class weights from loaded training data...")
    all_labels = []
    for ds in datasets_list:
        if hasattr(ds, 'can_labels'):
            all_labels.extend(ds.can_labels.tolist())
        elif hasattr(ds, 'eth_label'):
            all_labels.extend([ds.eth_label] * len(ds))

    if all_labels:
        from collections import Counter

        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        num_classes = max(label_counts.keys()) + 1
        class_weights = torch.zeros(num_classes, device=device)
        for cls, count in label_counts.items():
            class_weights[cls] = total / (num_classes * count)
        print(f"  Class distribution: {dict(label_counts)}")
        print(f"  Class weights: {class_weights.cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("  Warning: Could not compute class weights, using uniform.")
        criterion = nn.CrossEntropyLoss()

    full_ds = ConcatDataset(datasets_list)
    print(f"Total samples: {len(full_ds)}")
    
    # Split
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Training Loop
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                 xc = batch['can'].to(device)
                 xe = batch['eth'].to(device)
                 labels = batch['label'].to(device)
            else:
                 (xc, xe), labels = batch
                 xc = xc.to(device)
                 xe = xe.to(device)
                 labels = labels.to(device)
                 
            optimizer.zero_grad()
            logits = model(xc, xe)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
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
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {f1:.4f} | Acc: {acc:.4f}")
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(args.output_dir, "student_tiny_improved.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_rows", type=int, default=None, help="Limit rows for speed")
    parser.add_argument("--full_data", action="store_true", help="Use all pairs")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_light_model(args)

if __name__ == "__main__":
    main()
