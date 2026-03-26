import os
import sys
import argparse
from collections import Counter
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.data_resolvers import resolve_can_csv, resolve_eth_packet_csv
from src_replica.hybrid_curriculum import training_pair_specs
from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

def main():
    parser = argparse.ArgumentParser(description="Train Improved Heavy Model (Random Forest)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Data directory")
    parser.add_argument("--output_model", type=str, default="models/heavy_rf_improved.joblib", help="Output model path")
    parser.add_argument("--max_rows", type=int, default=None, help="Max rows per CAN file for quick testing")
    args = parser.parse_args()

    can_features = STANDARD_CAN_FEATURES_16

    pairs = [(spec.can_file, spec.eth_npy_file, spec) for spec in training_pair_specs()]

    datasets_with_specs = []
    loaded_pairs = set()
    
    for can_f, eth_n, spec in pairs:
        pair_key = (spec.name, can_f, eth_n)
        if pair_key in loaded_pairs: continue
        
        can_path = resolve_can_csv(args.data_dir, can_f, prefer_raw=True)
        eth_npy_path = os.path.join(args.data_dir, eth_n)
        eth_csv_path = resolve_eth_packet_csv(args.data_dir, eth_n)
        
        if can_path and os.path.exists(can_path) and eth_csv_path:
            print(f"Loading pair: {spec.name} [{spec.group}] -> {can_f} + {eth_n}")
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
                    datasets_with_specs.append((spec, ds))
                    loaded_pairs.add(pair_key)
                    print(f"  -> Added {len(ds)} samples.")
                else:
                    print(f"  -> Warning: Dataset empty after alignment.")
            except Exception as e:
                print(f"Error loading {can_f}: {e}")
        else:
            print(f"Skipping pair {can_f}, missing files.")

    if not datasets_with_specs:
        print("No datasets loaded. Exiting.")
        return

    total_samples = int(sum(len(ds) for _, ds in datasets_with_specs))
    print(f"Total samples: {total_samples}")
    
    print("Extracting features for Random Forest...")
    X_list = []
    y_list = []
    pair_name_list = []
    group_name_list = []

    processed = 0
    for spec, ds in datasets_with_specs:
        for i in range(len(ds)):
            data = ds[i]
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

            xc_flat = xc.flatten()
            xe_flat = xe.flatten()

            x_combined = np.concatenate([xc_flat, xe_flat])
            X_list.append(x_combined)
            y_list.append(label)
            pair_name_list.append(spec.name)
            group_name_list.append(spec.group)

            processed += 1
            if processed % 500 == 0:
                print(f"Processed {processed}/{total_samples} samples...")
            
    X = np.array(X_list)
    y = np.array(y_list)
    label_counts = Counter(int(label) for label in y.tolist())
    pair_counts = Counter(pair_name_list)
    group_counts = Counter(group_name_list)

    label_weight_map = {
        cls: float(len(y) / (max(len(label_counts), 1) * count))
        for cls, count in label_counts.items()
        if count > 0
    }
    pair_weight_map = {
        name: float(len(pair_name_list) / (max(len(pair_counts), 1) * count))
        for name, count in pair_counts.items()
        if count > 0
    }
    sample_weight = np.asarray(
        [label_weight_map[int(label)] * pair_weight_map[pair_name] for label, pair_name in zip(y, pair_name_list)],
        dtype=np.float64,
    )
    
    print(f"Training Random Forest on {X.shape[0]} samples with {X.shape[1]} features...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, class_weight='balanced', random_state=42)
    clf.fit(X, y, sample_weight=sample_weight)
    
    print(f"Training Accuracy: {clf.score(X, y):.4f}")
    
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(clf, args.output_model)
    print(f"Saved improved heavy model to {args.output_model}")

    report_path = os.path.splitext(args.output_model)[0] + "_train_report.json"
    report = {
        "model_path": args.output_model,
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "training_accuracy": float(clf.score(X, y)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "group_distribution": {str(k): int(v) for k, v in group_counts.items()},
        "pair_distribution": {str(k): int(v) for k, v in pair_counts.items()},
        "sample_weighting": {
            "strategy": "inverse_class_x_inverse_pair_frequency",
            "pair_weights": {str(k): float(v) for k, v in pair_weight_map.items()},
            "class_weights": {str(k): float(v) for k, v in label_weight_map.items()},
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2)
    print(f"Saved heavy training report to {report_path}")

if __name__ == "__main__":
    main()
