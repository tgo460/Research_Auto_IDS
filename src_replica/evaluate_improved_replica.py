import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Adjust path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from architecture_improved import TinyHybridStudent
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

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
        ("can_dos_train.csv", "eth_driving_01_injected_images-003.npy", "eth_driving_01_injected.csv"),
        ("can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy", "eth_driving_02_injected.csv"),
        ("can_gear_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"), 
        ("can_rpm_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
    ]
    
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

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    print("\n--- Improved Model Results ---")
    print(f"Accuracy:  {acc:.4f}")
    if len(set(all_labels)) > 1:
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
        print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
    else:
        print("Single class in evaluation set - skipping F1/CM.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--model_path", type=str, default="models/student_tiny_improved.pth")
    # Set default None to load full files
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_improved(args)

if __name__ == "__main__":
    main()
