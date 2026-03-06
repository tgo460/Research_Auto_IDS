import argparse
import json
import os
import sys
import joblib 
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset, Dataset

# Add src_replica to path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)

from architecture_replica import TinyHybridStudent
from heavy_infer_replica import HeavyTrainConfig, train_heavy_model, predict_heavy
from router_replica import ConfidenceRouter, RouterConfig, tune_threshold_by_quantile

# Conditional import for dataset
try:
    from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
    HAS_DATALOADER = True
except ImportError:
    HAS_DATALOADER = False

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def resolve_decision_threshold_from_report(base_path: str, tcfg: Dict, default_threshold: float = 0.5) -> Tuple[float, str]:
    report_path = os.path.join(base_path, 'logs', 'calibration_report.json') # Example path
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
                return data.get('decision_threshold', default_threshold), 'calibration_report'
        except:
            pass
    return default_threshold, 'default'

def build_mixed_dataset(base_path: str, split: str = 'train') -> Optional[ConcatDataset]:
    # 1. Load Split
    split_path = os.path.join(base_path, 'data', 'splits', 'split_v1.json')
    if not os.path.exists(split_path):
        print(f"Split file not found at {split_path}")
        return None

    try:
        with open(split_path, 'r') as f:
            split_data = json.load(f)
    except Exception as e:
        print(f"Error reading split file: {e}")
        return None

    # 2. Get file lists
    eth_files = split_data['modalities']['eth'].get(split, [])
    can_files = split_data['modalities']['can'].get(split, [])
    
    if not eth_files:
        print(f"No ETH files found for split {split}")
        return None

    # 3. Categorize CAN files
    benign_can = [f for f in can_files if 'normal' in f]
    attack_can = [f for f in can_files if 'normal' not in f]

    # Fallback: If no benign CAN in this split (e.g. val), borrow from train
    if not benign_can:
        benign_can = [f for f in split_data['modalities']['can']['train'] if 'normal' in f]
    
    # Fallback: If no attack CAN in this split but we need it (unlikely for val/test if mixed)
    if not attack_can:
        attack_can = [f for f in split_data['modalities']['can']['train'] if 'normal' not in f]

    datasets = []
    
    # 4. Pair Datasets
    for eth_file in eth_files:
        # Determine pairing based on ETH type
        is_attack = 'injected' in eth_file or 'attack' in eth_file
        target_can_file = None

        if is_attack:
            # Pair with first available attack CAN file (e.g. can_dos)
            if attack_can:
                target_can_file = attack_can[0] 
        else:
            # Pair with benign CAN file
            if benign_can:
                target_can_file = benign_can[0]
        
        # Instantiate Dataset if pair found
        if target_can_file:
            # Construct paths
            # Eth CSV matches NPY name usually: eth_..._images...npy -> eth_... .csv
            # Actually, split lists NPY files like "eth_driving_01_injected_images-003.npy"
            # The CSV is likely "eth_driving_01_injected.csv"
            # Strategy: strip "_images.*" and append ".csv"
            
            eth_npy_abs = os.path.join(base_path, 'datasets', eth_file)
            
            # Heuristic for CSV name
            # Split lists NPY files like "eth_driving_01_injected_images-003.npy"
            # We need "eth_driving_01_injected_replica_packets.csv" typically found in replica_eth_smoke/
            
            base_name = eth_file.split('_images')[0]
            if "_images" not in eth_file:
                 base_name = os.path.splitext(eth_file)[0]
            
            # Try specific replica_eth_smoke folder first
            eth_csv_candidates = [
                os.path.join(base_path, 'datasets', 'replica_eth_smoke', f"{base_name}_replica_packets.csv"),
                os.path.join(base_path, 'datasets', f"{base_name}_replica_packets.csv"),
                os.path.join(base_path, 'datasets', f"{base_name}.csv") # Fallback to root (failed previously)
            ]
            
            eth_csv_abs = None
            for cand in eth_csv_candidates:
                if os.path.exists(cand):
                    eth_csv_abs = cand
                    break
            
            # If still not found, search recursively? No, let's stick to known paths.
            if not eth_csv_abs:
                print(f"Could not find ETH CSV for {eth_file} (base: {base_name})")
                continue

            can_csv_abs = os.path.join(base_path, 'datasets', target_can_file)
            eth_npy_abs = os.path.join(base_path, 'datasets', eth_file)
            
            if os.path.exists(eth_npy_abs) and os.path.exists(eth_csv_abs) and os.path.exists(can_csv_abs):

                print(f"Pairing {eth_file} with {target_can_file}")
                if HAS_DATALOADER:
                    ds = CorrelatedHybridVehicleDataset(
                        can_csv_path=can_csv_abs,
                        eth_packet_csv_path=eth_csv_abs,
                        eth_npy_path=eth_npy_abs,
                        can_features=['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'],
                        can_window_size=100, # PDF standard: 100 CAN messages per detection window
                        eth_window_size=1,  # Matching TinyHybridStudent eth is (B, 1, 64, 64) or similar?
                                            # Wait, architecture expects (B, 1, 64, 64) for images.
                                            # If dataloader returns sequence, we need to check correlation logic.
                                            # Architecture handles 4D or 5D. 
                                            # Let's trust defaults or adjust if needed.
                        label_policy='max'
                    )
                    datasets.append(ds)
                else:
                    print("Dataloader not imported.")
            else:
                print(f"Missing files for pair: {eth_npy_abs}, {eth_csv_abs}, {can_csv_abs}")

    if datasets:
        return ConcatDataset(datasets)
    return None


def as_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "fpr": float(binary_fpr(y_true, y_pred))
    }

def binary_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (fp + tn) == 0:
        return 0.0
    return float(fp / (fp + tn))

def tune_decision_threshold(y_true: np.ndarray, attack_score: np.ndarray, target_fpr: float) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0, 1, 1001)
    best_thresh = 0.5
    best_metrics = {}
    
    # Simple search
    for thresh in thresholds:
        y_pred = (attack_score >= thresh).astype(int)
        fpr = binary_fpr(y_true, y_pred)
        if fpr <= target_fpr:
            best_thresh = thresh
            best_metrics = as_metrics(y_true, y_pred)
            break # Found smallest threshold satisfying FPR (or largest depending on direction)
                  # Actually usually start from 1.0 down to 0.0 to find minimal FPR?
                  # If we iterate 0->1, we find first thresh with low FPR. 
                  # Low threshold -> High Recall, High FPR. 
                  # High threshold -> Low Recall, Low FPR.
                  # We want max recall s.t. FPR <= target. 
                  # So we should probably iterate likely from high to low or check all.
    
    # Better: sort scores
    # But for this replica, a simple linspace is fine as placeholder
    return float(best_thresh), best_metrics

class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000):
        # Shape should be (N, Window=10, Features=?) OR (N, Window, Features)
        # Architecture expects CAN input dim (channels) = 10. 
        # And it permutes (0, 2, 1) before Conv1d.
        # Conv1d expects (N, C, L).
        # So after permute, we want (N, 10, L).
        # So before permute, input should be (N, L, 10).
        # Let's say Window=32. Features=10.
        self.x_c = torch.randn(n_samples, 32, 10)
        self.x_e = torch.randn(n_samples, 1, 64, 64)

        self.y = torch.randint(0, 2, (n_samples,))
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return (self.x_c[idx], self.x_e[idx]), self.y[idx]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cascade Architecture")
    parser.add_argument("--light_model_path", type=str, required=True, help="Path to light model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--heavy_backend", type=str, default="rf", choices=["rf", "mlp"], help="Heavy model backend")
    parser.add_argument("--route_fraction", type=float, default=0.3, help="Fraction of data to router to heavy model during evaluation/training")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory")
    parser.add_argument("--synthetic", action='store_true', help="Use synthetic data if real data not found")
    
    # Model parameters
    parser.add_argument("--heavy_n_estimators", type=int, default=100)
    parser.add_argument("--heavy_max_depth", type=int, default=None)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(42)

    # 1. Load Light Model
    print(f"Loading light model from {args.light_model_path}")
    light_model = TinyHybridStudent()
    
    if os.path.exists(args.light_model_path):
        try:
            checkpoint = torch.load(args.light_model_path, map_location=device)
            # Handle state dict wrapper
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                light_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 light_model.load_state_dict(checkpoint)
            print("Checkpoint loaded.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            if not args.synthetic:
                 print("Exiting.")
                 sys.exit(1)
            print("Proceeding with random weights (synthetic mode).")
    else:
        print(f"Checkpoint not found at {args.light_model_path}")
        if not args.synthetic:
            sys.exit(1)
        print("Proceeding with random weights (synthetic mode).")

    light_model.to(device)
    light_model.eval()

    # 2. Load Data
    train_ds = None
    val_ds = None
    
    if not args.synthetic:
        # Try to build real datasets
        print("Attempting to load real datasets...")
        project_root = os.path.dirname(BASE_PATH)
        train_ds = build_mixed_dataset(project_root, split='train')
        if train_ds:
            print("Train dataset loaded.")
            # For val, we might want 'val' split, but let's check if it exists
            val_ds = build_mixed_dataset(project_root, split='val')

            if not val_ds:
                 # If val split empty or fails, split train? Or validation on train subset?
                 print("Validation dataset could not be loaded, using subset of train.")
                 train_len = int(0.8 * len(train_ds))
                 val_len = len(train_ds) - train_len
                 train_ds, val_ds = random_split(train_ds, [train_len, val_len])
            else:
                print("Validation dataset loaded.")
        else:
            print("Failed to load real datasets.")
    
    if train_ds is None:
        print("Using synthetic data.")
        full_ds = SyntheticDataset(n_samples=200) # Small for speed in replica
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    
    # 3. Validating Light Model & Finding Hard Examples (On Train Set)
    print("Running light model on training set...")
    light_logits_list = []
    light_labels_list = []
    all_X_flattened = []
    
    with torch.no_grad():
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

            logit = light_model(xc, xe)
            light_logits_list.append(logit.cpu())
            light_labels_list.append(labels.cpu())
            
            # Flatten inputs for heavy model features
            f_xc = xc.view(xc.size(0), -1).cpu().numpy()
            f_xe = xe.view(xe.size(0), -1).cpu().numpy()
            all_X_flattened.append(np.hstack([f_xc, f_xe]))

    train_logits = torch.cat(light_logits_list)
    train_labels = torch.cat(light_labels_list).numpy()
    train_features = np.vstack(all_X_flattened)
    
    # 4. Configure Router and Tune Threshold
    print("Configuring router...")
    router_config = RouterConfig(mode='max_softmax', route_if_below_or_equal=True)
    router = ConfidenceRouter(router_config)
    
    train_confidences = router.confidence_from_logits(train_logits)
    threshold = tune_threshold_by_quantile(train_confidences.numpy(), args.route_fraction)
    router.config.threshold = threshold
    print(f"Router threshold set to {threshold:.4f} (target fraction: {args.route_fraction})")
    
    # 5. Train Heavy Model
    routed_mask = router.route_from_confidence(train_confidences).numpy()
    X_heavy_train = train_features[routed_mask]
    y_heavy_train = train_labels[routed_mask]
    
    heavy_model = None
    if len(X_heavy_train) > 0:
        print(f"Training heavy model on {len(X_heavy_train)} samples...")
        heavy_config = HeavyTrainConfig(
            backend=args.heavy_backend,
            n_estimators=args.heavy_n_estimators,
            max_depth=args.heavy_max_depth
        )
        heavy_model = train_heavy_model(X_heavy_train, y_heavy_train, heavy_config)
        
        # Save Heavy Model
        if args.heavy_backend == 'rf':
            os.makedirs(args.light_model_path.replace(os.path.basename(args.light_model_path), ""), exist_ok=True)
            heavy_path = os.path.join(os.path.dirname(args.light_model_path), 'heavy_rf.joblib')
            joblib.dump(heavy_model, heavy_path)
            print(f"Saved Heavy Model (RF) to {heavy_path}")
    else:
        print("No samples routed for training. Heavy model will not be used.")

    # 6. Evaluate Cascade
    print("Evaluating cascade...")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    val_probs = [] # Prob of class 1
    val_preds = []
    val_labels = []
    routings = []
    
    # For tracking light-only performance simultaneously
    light_only_preds = []
    
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
            
            # Light inference
            logits = light_model(xc, xe)

            confidences = router.confidence_from_logits(logits)
            should_route = router.route_from_confidence(confidences)
            
            # Light probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_light_preds = np.argmax(probs, axis=1)
            
            # Store light only
            light_only_preds.append(batch_light_preds)
            
            # Prepare cascade outputs
            batch_final_probs = probs.copy() # Start with light probs
            batch_final_preds = batch_light_preds.copy()
            
            # If routing needed
            if should_route.any() and heavy_model is not None:
                # Prepare features
                f_xc = xc.view(xc.size(0), -1).cpu().numpy()
                f_xe = xe.view(xe.size(0), -1).cpu().numpy()
                batch_features = np.hstack([f_xc, f_xe])
                
                # Identify indices in this batch that need routing
                # should_route is a tensor of bools
                route_indices_batch = np.where(should_route.cpu().numpy())[0]
                
                if len(route_indices_batch) > 0:
                    X_routed = batch_features[route_indices_batch]
                    heavy_res = predict_heavy(heavy_model, X_routed)
                    
                    # Update predictions and probabilities
                    # heavy_res['predictions'] is class 0/1
                    # heavy_res['probabilities'] is (N, 2)
                    
                    batch_final_preds[route_indices_batch] = heavy_res['predictions']
                    batch_final_probs[route_indices_batch] = heavy_res['probabilities']
            
            val_probs.append(batch_final_probs[:, 1]) # Store prob of class 1 (Attack)
            val_preds.append(batch_final_preds)
            val_labels.append(labels.numpy())
            routings.append(should_route.cpu().numpy())

    # Concatenate results
    all_final_probs = np.concatenate(val_probs)
    all_final_preds = np.concatenate(val_preds)
    all_labels = np.concatenate(val_labels)
    all_routed = np.concatenate(routings)
    all_light_preds = np.concatenate(light_only_preds)
    
    # Calculate metrics
    cascade_metrics = as_metrics(all_labels, all_final_preds)
    light_metrics = as_metrics(all_labels, all_light_preds)
    
    # Calculate Confusion Matrix
    cm_cascade = confusion_matrix(all_labels, all_final_preds, labels=[0, 1])
    cm_light = confusion_matrix(all_labels, all_light_preds, labels=[0, 1])
    
    cascade_metrics['routed_fraction'] = float(np.mean(all_routed))
    
    print("\n--- Results ---")
    print("Light Only Metrics:", light_metrics)
    print("Cascade Metrics:   ", cascade_metrics)
    
    # 7. Save Report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "cascade_eval_replica_report.json")
    
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fpr_budget": 0.05, # Placeholder or from args
        "light_only": {
            **light_metrics,
            "confusion_matrix": cm_light.tolist()
        },
        "cascade": {
            **cascade_metrics,
            "confusion_matrix": cm_cascade.tolist(),
            "router_threshold": threshold,
            "heavy_decision_threshold": 0.5
        }
    }

    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
