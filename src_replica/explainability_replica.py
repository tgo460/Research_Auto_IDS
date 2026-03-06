import os
import argparse
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset

from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

def main():
    parser = argparse.ArgumentParser(description="Explainability for Heavy Model using SHAP")
    parser.add_argument("--model_path", type=str, default="models/heavy_rf_improved.joblib", help="Path to heavy model")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Data directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of background samples for SHAP")
    args = parser.parse_args()

    print(f"Loading Heavy Model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    model = joblib.load(args.model_path)

    can_features = [
        'CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
        'can_id_freq_global', 'can_id_freq_win', 'payload_entropy', 
        'inter_arrival', 'inter_arrival_roll_mean', 'id_switch_rate_win'
    ]

    # Generate feature names
    feature_names = []
    for w in range(32):
        for f in can_features:
            feature_names.append(f"CAN_W{w}_{f}")
            
    for p in range(1024):
        feature_names.append(f"ETH_Px{p}")

    print("Loading a small dataset to compute SHAP values...")
    engineered_dir = os.path.join(args.data_dir, "replica_can_b1_engineered")
    
    # Just load one file for background data
    can_f = "can_fuzzy_train.csv"
    eth_n = "eth_driving_02_injected_images-008.npy"
    eth_c = "eth_driving_02_injected_replica_packets.csv"
    
    can_path = os.path.join(engineered_dir, can_f)
    eth_npy_path = os.path.join(args.data_dir, eth_n)
    eth_csv_path = os.path.join(args.data_dir, "replica_eth_smoke", eth_c)
    
    if not os.path.exists(eth_csv_path):
        eth_csv_path = os.path.join(args.data_dir, eth_c)
        
    try:
        ds = CorrelatedHybridVehicleDataset(
            can_csv_path=can_path,
            eth_packet_csv_path=eth_csv_path,
            eth_npy_path=eth_npy_path,
            can_features=can_features,
            can_window_size=CAN_WINDOW_SIZE_STANDARD,
            eth_window_size=ETH_WINDOW_SIZE_STANDARD,
            can_max_rows=500 # Just need a few
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    X_list = []
    for i in range(min(args.num_samples, len(ds))):
        data = ds[i]
        if isinstance(data, tuple):
             if len(data) == 2:
                 (xc, xe), label = data
             else:
                 xc, xe, label = data
        elif isinstance(data, dict):
             xc = data['can']
             xe = data['eth']
             
        xc_flat = xc.flatten()
        xe_flat = xe.flatten()
        x_combined = np.concatenate([xc_flat, xe_flat])
        X_list.append(x_combined)
        
    X_bg = np.array(X_list)
    
    print(f"Computing SHAP values for {X_bg.shape[0]} samples...")
    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_bg)
    
    # For binary classification, shap_values is a list of length 2. We want the values for class 1 (Malicious)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        # In newer SHAP versions, it might be an Explanation object or 3D array
        if len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values
            
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_class1, X_bg, feature_names=feature_names, show=False, max_display=20)
    
    output_plot = "reports/shap_summary_improved.png"
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved SHAP summary plot to {output_plot}")

if __name__ == "__main__":
    main()
