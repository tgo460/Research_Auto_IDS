import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import ConcatDataset
import torch

from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

def main():
    parser = argparse.ArgumentParser(description="Train Improved Heavy Model (Random Forest)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Data directory")
    parser.add_argument("--output_model", type=str, default="models/heavy_rf_improved.joblib", help="Output model path")
    parser.add_argument("--max_rows", type=int, default=None, help="Max rows per CAN file for quick testing")
    args = parser.parse_args()

    can_features = [
        'CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
        'can_id_freq_global', 'can_id_freq_win', 'payload_entropy', 
        'inter_arrival', 'inter_arrival_roll_mean', 'id_switch_rate_win'
    ]

    engineered_dir = os.path.join(args.data_dir, "replica_can_b1_engineered")
    
    pairs = [
        ("can_dos_train.csv", "eth_driving_01_injected_images-003.npy", "eth_driving_01_injected.csv"),
        ("can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy", "eth_driving_02_injected.csv"),
        ("can_gear_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
        ("can_rpm_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
    ]
    
    datasets_list = []
    loaded_files = set()
    
    for can_f, eth_n, eth_c in pairs:
        if can_f in loaded_files: continue
        
        can_path = os.path.join(engineered_dir, can_f)
        eth_npy_path = os.path.join(args.data_dir, eth_n)
        
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
                    can_max_rows=args.max_rows
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
            print(f"Skipping pair {can_f}, missing files.")

    if not datasets_list:
        print("No datasets loaded. Exiting.")
        return

    full_ds = ConcatDataset(datasets_list)
    print(f"Total samples: {len(full_ds)}")
    
    print("Extracting features for Random Forest...")
    X_list = []
    y_list = []
    
    for i in range(len(full_ds)):
        data = full_ds[i]
        if isinstance(data, tuple):
             if len(data) == 2:
                 (xc, xe), label = data
             else:
                 xc, xe, label = data
        elif isinstance(data, dict):
             xc = data['can']
             xe = data['eth']
             label = data['label']
             
        if isinstance(label, torch.Tensor):
            label = int(label.item())
            
        # Flatten CAN (32, 16) -> 512
        xc_flat = xc.flatten()
        # Flatten ETH (1, 1, 32, 32) -> 1024
        xe_flat = xe.flatten()
        
        x_combined = np.concatenate([xc_flat, xe_flat])
        X_list.append(x_combined)
        y_list.append(label)
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(full_ds)} samples...")
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Training Random Forest on {X.shape[0]} samples with {X.shape[1]} features...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    
    print(f"Training Accuracy: {clf.score(X, y):.4f}")
    
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(clf, args.output_model)
    print(f"Saved improved heavy model to {args.output_model}")

if __name__ == "__main__":
    main()
