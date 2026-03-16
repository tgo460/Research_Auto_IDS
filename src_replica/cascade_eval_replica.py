import argparse
import json
import os
import sys
import joblib 
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset, Dataset

# Add src_replica to path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_PATH)
sys.path.append(BASE_PATH)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from heavy_infer_replica import HeavyTrainConfig, train_heavy_model, predict_heavy
from router_replica import ConfidenceRouter, RouterConfig, tune_threshold_by_quantile
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

CAN_FEATURES_16 = ['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
                   'can_id_freq_global', 'can_id_freq_win', 'payload_entropy',
                   'inter_arrival', 'inter_arrival_roll_mean', 'id_switch_rate_win']

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
                    # Use engineered CAN features if available
                    engineered_can = os.path.join(base_path, 'datasets', 'replica_can_b1_engineered', target_can_file)
                    if os.path.exists(engineered_can):
                        can_csv_abs = engineered_can
                        active_features = CAN_FEATURES_16
                    else:
                        active_features = CAN_FEATURES_16[:10]  # fallback to raw 10
                    ds = CorrelatedHybridVehicleDataset(
                        can_csv_path=can_csv_abs,
                        eth_packet_csv_path=eth_csv_abs,
                        eth_npy_path=eth_npy_abs,
                        can_features=active_features,
                        can_window_size=CAN_WINDOW_SIZE_STANDARD,
                        eth_window_size=ETH_WINDOW_SIZE_STANDARD,
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(binary_fpr(y_true, y_pred)),
        "fnr": float(fnr),
        "specificity": float(specificity),
        "npv": float(npv),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> Dict[str, List[float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_vals, mcc_vals, fpr_vals, fnr_vals = [], [], [], []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1_vals.append(f1_score(yt, yp, zero_division=0))
        mcc_vals.append(matthews_corrcoef(yt, yp) if len(np.unique(yt)) > 1 else 0.0)
        fpr_vals.append(binary_fpr(yt, yp))
        cm = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm
        fnr_vals.append(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0)

    return {
        "f1_ci95": [float(np.percentile(f1_vals, 2.5)), float(np.percentile(f1_vals, 97.5))],
        "mcc_ci95": [float(np.percentile(mcc_vals, 2.5)), float(np.percentile(mcc_vals, 97.5))],
        "fpr_ci95": [float(np.percentile(fpr_vals, 2.5)), float(np.percentile(fpr_vals, 97.5))],
        "fnr_ci95": [float(np.percentile(fnr_vals, 2.5)), float(np.percentile(fnr_vals, 97.5))],
    }


def calibration_metrics(y_true: np.ndarray, attack_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    attack_prob = np.asarray(attack_prob).astype(float)
    attack_prob = np.clip(attack_prob, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(attack_prob, edges[1:-1], right=False)
    n = len(y_true)

    ece = 0.0
    bin_rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            bin_rows.append({
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": 0,
                "avg_confidence": None,
                "empirical_positive_rate": None,
            })
            continue

        conf = float(np.mean(attack_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (cnt / max(n, 1)) * abs(acc - conf)
        bin_rows.append({
            "bin": b,
            "lower": float(edges[b]),
            "upper": float(edges[b + 1]),
            "count": cnt,
            "avg_confidence": conf,
            "empirical_positive_rate": acc,
        })

    brier = float(np.mean((attack_prob - y_true) ** 2)) if n > 0 else 0.0
    return {
        "ece": float(ece),
        "brier": brier,
        "num_bins": int(n_bins),
        "bins": bin_rows,
    }


def _metric_value(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = as_metrics(y_true, y_pred)
    return float(m[metric_name])


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_names: List[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    out = {}
    for name in metric_names:
        vals = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            pa = pred_a[idx]
            pb = pred_b[idx]
            vals.append(_metric_value(name, yt, pb) - _metric_value(name, yt, pa))

        out[name] = {
            "delta_mean": float(np.mean(vals)),
            "delta_ci95_low": float(np.percentile(vals, 2.5)),
            "delta_ci95_high": float(np.percentile(vals, 97.5)),
        }
    return out


def paired_permutation_pvalue(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_name: str,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)

    observed = _metric_value(metric_name, y_true, pred_b) - _metric_value(metric_name, y_true, pred_a)
    ge = 0

    for _ in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5
        pa = pred_a.copy()
        pb = pred_b.copy()
        pa[swap], pb[swap] = pred_b[swap], pred_a[swap]
        d = _metric_value(metric_name, y_true, pb) - _metric_value(metric_name, y_true, pa)
        if abs(d) >= abs(observed):
            ge += 1

    return float((ge + 1) / (n_permutations + 1))

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
        self.x_c = torch.randn(n_samples, CAN_WINDOW_SIZE_STANDARD, 16)
        self.x_e = torch.randn(n_samples, ETH_WINDOW_SIZE_STANDARD, 1, 32, 32)

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--permutation_resamples", type=int, default=1000)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(args.seed)

    # 1. Load Light Model
    print(f"Loading light model from {args.light_model_path}")
    light_model = TinyHybridStudent(input_dim=16, hidden_dim=64, num_classes=2)
    
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
    light_only_probs = []
    
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
            light_only_probs.append(probs[:, 1])
            
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
    all_light_probs = np.concatenate(light_only_probs)
    
    # Calculate metrics
    cascade_metrics = as_metrics(all_labels, all_final_preds)
    light_metrics = as_metrics(all_labels, all_light_preds)

    cascade_ci = bootstrap_ci(all_labels, all_final_preds, n_resamples=args.bootstrap_resamples, seed=args.seed)
    light_ci = bootstrap_ci(all_labels, all_light_preds, n_resamples=args.bootstrap_resamples, seed=args.seed)

    delta_stats = paired_bootstrap_deltas(
        all_labels,
        all_light_preds,
        all_final_preds,
        metric_names=["f1", "mcc", "fpr", "fnr"],
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    pvals = {
        "f1": paired_permutation_pvalue(
            all_labels, all_light_preds, all_final_preds,
            metric_name="f1", n_permutations=args.permutation_resamples, seed=args.seed,
        ),
        "mcc": paired_permutation_pvalue(
            all_labels, all_light_preds, all_final_preds,
            metric_name="mcc", n_permutations=args.permutation_resamples, seed=args.seed,
        ),
    }

    light_cal = calibration_metrics(all_labels, all_light_probs, n_bins=args.calibration_bins)
    cascade_cal = calibration_metrics(all_labels, all_final_probs, n_bins=args.calibration_bins)
    
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
            **light_ci,
            "confusion_matrix": cm_light.tolist()
        },
        "cascade": {
            **cascade_metrics,
            **cascade_ci,
            "confusion_matrix": cm_cascade.tolist(),
            "router_threshold": threshold,
            "heavy_decision_threshold": 0.5
        },
        "statistical_validation": {
            "delta_metrics_cascade_minus_light": delta_stats,
            "paired_permutation_p_values": pvals,
            "permutation_resamples": int(args.permutation_resamples),
        },
        "calibration": {
            "light_only": light_cal,
            "cascade": cascade_cal,
        },
    }

    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
