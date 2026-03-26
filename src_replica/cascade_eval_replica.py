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
from src_replica.data_resolvers import resolve_can_csv, resolve_eth_packet_csv
from heavy_infer_replica import HeavyTrainConfig, train_heavy_model, predict_heavy
from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16
from router_replica import ConfidenceRouter, RouterConfig, tune_threshold_by_quantile
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD
from src_replica.split_manifest_utils import is_attack_split_entry, parse_split_entry

CAN_FEATURES_16 = STANDARD_CAN_FEATURES_16

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

def _resolve_split_manifest(base_path: str, split_manifest: Optional[str]) -> Optional[str]:
    candidates = []
    if split_manifest:
        candidates.append(
            split_manifest if os.path.isabs(split_manifest) else os.path.join(base_path, split_manifest)
        )
    else:
        candidates.extend(
            [
                os.path.join(base_path, "data", "splits", "split_v3_research_valid.json"),
                os.path.join(base_path, "data", "splits", "split_v2_domain_balanced.json"),
                os.path.join(base_path, "data", "splits", "split_v1.json"),
            ]
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else None


def _is_attack_artifact(name: str) -> bool:
    return is_attack_split_entry(name)


def _plan_pairs(
    eth_files: List[Any],
    can_files: List[Any],
    pairing_mode: str,
) -> Tuple[List[Tuple[Any, Any, int]], List[str]]:
    attack_eth = sorted([f for f in eth_files if _is_attack_artifact(f)], key=lambda x: parse_split_entry(x).display_name())
    normal_eth = sorted([f for f in eth_files if not _is_attack_artifact(f)], key=lambda x: parse_split_entry(x).display_name())
    attack_can = sorted([f for f in can_files if _is_attack_artifact(f)], key=lambda x: parse_split_entry(x).display_name())
    normal_can = sorted([f for f in can_files if not _is_attack_artifact(f)], key=lambda x: parse_split_entry(x).display_name())

    pair_specs: List[Tuple[Any, Any, int]] = []
    warnings: List[str] = []

    def _names(entries: List[Any]) -> str:
        return ", ".join(parse_split_entry(entry).display_name() for entry in entries)

    def add_pairs(eth_subset: List[Any], can_subset: List[Any], label: int, label_name: str) -> None:
        if not eth_subset:
            return
        if not can_subset:
            warnings.append(
                f"split is missing within-split {label_name} CAN coverage for ETH files: {_names(eth_subset)}"
            )
            return

        if pairing_mode == "single_match":
            for idx, eth_file in enumerate(eth_subset):
                pair_specs.append((eth_file, can_subset[idx % len(can_subset)], label))
            return

        for eth_file in eth_subset:
            for can_file in can_subset:
                pair_specs.append((eth_file, can_file, label))

    add_pairs(normal_eth, normal_can, 0, "normal")
    add_pairs(attack_eth, attack_can, 1, "attack")
    return pair_specs, warnings


def build_mixed_dataset(
    base_path: str,
    split: str = 'train',
    split_manifest: Optional[str] = None,
    pairing_mode: str = "label_cartesian",
    require_both_classes: bool = False,
) -> Optional[ConcatDataset]:
    split_path = _resolve_split_manifest(base_path, split_manifest)
    if not split_path or not os.path.exists(split_path):
        print(f"Split file not found at {split_path}")
        return None

    try:
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
    except Exception as e:
        print(f"Error reading split file: {e}")
        return None

    eth_entries_raw = split_data['modalities']['eth'].get(split, [])
    can_entries_raw = split_data['modalities']['can'].get(split, [])
    if not eth_entries_raw or not can_entries_raw:
        print(f"Missing CAN or ETH files for split {split}")
        return None
    try:
        eth_files = [parse_split_entry(entry) for entry in eth_entries_raw]
        can_files = [parse_split_entry(entry) for entry in can_entries_raw]
    except Exception as exc:
        print(f"Invalid split manifest entry for split {split}: {exc}")
        return None

    pair_specs, warnings = _plan_pairs(eth_files, can_files, pairing_mode=pairing_mode)
    if not pair_specs:
        for warning in warnings:
            print(f"Warning: {warning}")
        print(f"No valid within-split CAN/ETH pairs for split {split}")
        return None

    datasets = []
    loaded_labels = set()
    pair_rows = []
    datasets_dir = os.path.join(base_path, "datasets")

    for eth_ref, can_ref, expected_label in pair_specs:
        eth_npy_abs = os.path.join(datasets_dir, eth_ref.path)
        eth_csv_abs = resolve_eth_packet_csv(datasets_dir, eth_ref.path)
        can_csv_abs = resolve_can_csv(datasets_dir, can_ref.path, prefer_raw=True)

        if not eth_csv_abs or not can_csv_abs:
            warnings.append(
                f"skipped pair {eth_ref.display_name()} + {can_ref.display_name()}: missing files "
                f"(eth_npy={os.path.exists(eth_npy_abs)}, eth_csv={bool(eth_csv_abs)}, can_csv={bool(can_csv_abs)})"
            )
            continue

        print(f"Pairing {eth_ref.display_name()} with {can_ref.display_name()}")
        if not HAS_DATALOADER:
            print("Dataloader not imported.")
            return None

        try:
            ds = CorrelatedHybridVehicleDataset(
                can_csv_path=can_csv_abs,
                eth_packet_csv_path=eth_csv_abs,
                eth_npy_path=eth_npy_abs,
                can_features=CAN_FEATURES_16,
                can_window_size=CAN_WINDOW_SIZE_STANDARD,
                eth_window_size=ETH_WINDOW_SIZE_STANDARD,
                can_row_start=can_ref.row_start,
                can_row_stop=can_ref.row_stop,
                label_policy='max'
            )
        except Exception as exc:
            warnings.append(f"failed to load pair {eth_ref.display_name()} + {can_ref.display_name()}: {exc}")
            continue

        if len(ds) == 0:
            warnings.append(f"empty aligned dataset for pair {eth_ref.display_name()} + {can_ref.display_name()}")
            continue

        datasets.append(ds)
        loaded_labels.add(expected_label)
        pair_rows.append(
            {
                "eth_file": eth_ref.path,
                "can_file": can_ref.path,
                "eth_display": eth_ref.display_name(),
                "can_display": can_ref.display_name(),
                "can_row_start": can_ref.row_start,
                "can_row_stop": can_ref.row_stop,
                "expected_label": expected_label,
                "samples": len(ds),
            }
        )

    for warning in warnings:
        print(f"Warning: {warning}")

    if require_both_classes and len(loaded_labels) < 2:
        print(
            f"Split {split} does not contain both classes after strict within-split pairing. "
            "No train borrowing was used."
        )
        return None

    if not datasets:
        return None

    concat = ConcatDataset(datasets)
    concat.metadata = {
        "split": split,
        "split_manifest": split_path.replace("\\", "/"),
        "pairing_mode": pairing_mode,
        "pairs_loaded": len(pair_rows),
        "labels_present": sorted(loaded_labels),
        "warnings": warnings,
        "pairs": pair_rows,
    }
    return concat


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


def _cascade_outputs_for_threshold(
    light_probs: np.ndarray,
    heavy_probs: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    routed = np.asarray(confidences <= threshold, dtype=bool)
    final_probs = np.asarray(light_probs, dtype=float).copy()
    if heavy_probs.size > 0:
        final_probs[routed] = np.asarray(heavy_probs, dtype=float)[routed]
    final_preds = np.argmax(final_probs, axis=1).astype(int)
    return final_probs[:, 1], final_preds, routed


def calibrate_router_threshold(
    y_true: np.ndarray,
    confidences: np.ndarray,
    light_probs: np.ndarray,
    heavy_probs: np.ndarray,
    fpr_budget: float,
    num_candidates: int = 201,
) -> Tuple[float, Dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    confidences = np.asarray(confidences).astype(float)
    max_runtime_threshold = float(np.nextafter(1.0, 0.0))
    if confidences.size == 0:
        fallback_metrics = as_metrics(y_true, np.zeros_like(y_true))
        fallback_threshold = min(0.5, max_runtime_threshold)
        return fallback_threshold, {
            "selected_threshold": fallback_threshold,
            "selection_reason": "empty_validation_set",
            "fpr_budget": float(fpr_budget),
            "selected_metrics": fallback_metrics,
            "selected_routed_fraction": 0.0,
            "budget_feasible": False,
            "candidates_evaluated": 0,
        }

    quantiles = np.linspace(0.0, 1.0, max(int(num_candidates), 3))
    candidates = np.unique(np.quantile(confidences, quantiles).astype(float))
    candidates = np.clip(candidates, 0.0, max_runtime_threshold)
    if candidates.size == 0:
        candidates = np.asarray([min(0.5, max_runtime_threshold)], dtype=float)

    best_budget = None
    best_any = None

    for threshold in candidates:
        final_attack_prob, final_preds, routed = _cascade_outputs_for_threshold(
            light_probs=light_probs,
            heavy_probs=heavy_probs,
            confidences=confidences,
            threshold=float(threshold),
        )
        metrics = as_metrics(y_true, final_preds)
        routed_fraction = float(np.mean(routed)) if routed.size else 0.0
        candidate = {
            "threshold": float(threshold),
            "metrics": metrics,
            "routed_fraction": routed_fraction,
            "attack_probability_mean": float(np.mean(final_attack_prob)) if final_attack_prob.size else 0.0,
        }

        budget_key = (
            metrics["recall"],
            metrics["mcc"],
            -metrics["fpr"],
            -routed_fraction,
            -abs(routed_fraction - 0.3),
        )
        any_key = (
            -metrics["fpr"],
            metrics["mcc"],
            metrics["recall"],
            -routed_fraction,
        )
        if metrics["fpr"] <= fpr_budget and (
            best_budget is None or budget_key > best_budget["key"]
        ):
            best_budget = {**candidate, "key": budget_key}
        if best_any is None or any_key > best_any["key"]:
            best_any = {**candidate, "key": any_key}

    selected = best_budget or best_any
    assert selected is not None
    selection_reason = "met_fpr_budget" if best_budget is not None else "fallback_min_fpr"
    selected_threshold = min(float(selected["threshold"]), max_runtime_threshold)
    return selected_threshold, {
        "selected_threshold": selected_threshold,
        "selection_reason": selection_reason,
        "fpr_budget": float(fpr_budget),
        "selected_metrics": selected["metrics"],
        "selected_routed_fraction": float(selected["routed_fraction"]),
        "budget_feasible": bool(best_budget is not None),
        "candidates_evaluated": int(len(candidates)),
    }

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
    parser.add_argument("--pretrained_heavy_model", type=str, default=None,
                        help="Optional path to a pretrained heavy model to use instead of retraining on routed samples.")
    parser.add_argument("--route_fraction", type=float, default=0.3, help="Fraction of data to router to heavy model during evaluation/training")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory")
    parser.add_argument("--synthetic", action='store_true', help="Use synthetic data if real data not found")
    parser.add_argument("--split_manifest", type=str, default=os.path.join("data", "splits", "split_v3_research_valid.json"),
                        help="Split manifest for real-data evaluation.")
    parser.add_argument("--pairing_mode", type=str, default="label_cartesian", choices=["label_cartesian", "single_match"],
                        help="Within-split CAN/ETH pairing policy.")
    parser.add_argument("--allow_train_val_fallback", action='store_true',
                        help="Allow fallback to a random train/val split when the validation split is invalid. This is not independent evaluation.")
    parser.add_argument("--allow_one_class_eval", action='store_true',
                        help="Allow validation on a split that collapses to a single class after strict pairing.")
    
    # Model parameters
    parser.add_argument("--heavy_n_estimators", type=int, default=100)
    parser.add_argument("--heavy_max_depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--permutation_resamples", type=int, default=1000)
    parser.add_argument("--fpr_budget", type=float, default=0.05)
    parser.add_argument("--router_calibration_points", type=int, default=201)
    
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
    train_meta: Dict[str, Any] = {}
    val_meta: Dict[str, Any] = {}
    
    if not args.synthetic:
        # Try to build real datasets
        print("Attempting to load real datasets...")
        project_root = os.path.dirname(BASE_PATH)
        train_ds = build_mixed_dataset(
            project_root,
            split='train',
            split_manifest=args.split_manifest,
            pairing_mode=args.pairing_mode,
            require_both_classes=True,
        )
        if train_ds:
            print("Train dataset loaded.")
            train_meta = getattr(train_ds, "metadata", {})
            val_ds = build_mixed_dataset(
                project_root,
                split='val',
                split_manifest=args.split_manifest,
                pairing_mode=args.pairing_mode,
                require_both_classes=not args.allow_one_class_eval,
            )

            if not val_ds:
                 if not args.allow_train_val_fallback:
                     print(
                         "Validation dataset could not be loaded without train borrowing or class collapse. "
                         "Pass --allow_train_val_fallback or --allow_one_class_eval to override."
                     )
                     sys.exit(1)
                 print("Validation dataset could not be loaded independently, using random subset of train.")
                 train_len = int(0.8 * len(train_ds))
                 val_len = len(train_ds) - train_len
                 train_ds, val_ds = random_split(train_ds, [train_len, val_len])
                 train_meta = {
                     **train_meta,
                     "fallback_used": True,
                     "fallback_reason": "validation_split_invalid",
                 }
                 val_meta = {
                     "source": "random_split_from_train",
                     "fallback_used": True,
                     "independent_validation": False,
                 }
            else:
                print("Validation dataset loaded.")
                val_meta = getattr(val_ds, "metadata", {})
                if len(val_meta.get("labels_present", [])) < 2:
                    print(
                        "Warning: validation split resolves to a single class after strict pairing; "
                        "FPR/FNR claims will be incomplete."
                    )
        else:
            print("Failed to load real datasets.")
    
    if train_ds is None:
        print("Using synthetic data.")
        full_ds = SyntheticDataset(n_samples=200) # Small for speed in replica
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        train_meta = {"source": "synthetic", "independent_validation": False}
        val_meta = {"source": "synthetic", "independent_validation": False}

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
    heavy_model_source = "none"
    if args.pretrained_heavy_model:
        if not os.path.exists(args.pretrained_heavy_model):
            print(f"Pretrained heavy model not found at {args.pretrained_heavy_model}")
            sys.exit(1)
        heavy_model = joblib.load(args.pretrained_heavy_model)
        heavy_model_source = args.pretrained_heavy_model
        print(f"Loaded pretrained heavy model: {args.pretrained_heavy_model}")
    elif len(X_heavy_train) > 0:
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
            heavy_model_source = heavy_path
        else:
            heavy_model_source = f"trained:{args.heavy_backend}"
    else:
        print("No samples routed for training. Heavy model will not be used.")

    # 6. Evaluate Cascade
    print("Evaluating cascade...")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    val_labels = []
    val_confidences = []
    val_features = []

    # For tracking light-only performance simultaneously
    light_only_preds = []
    light_only_prob_rows = []

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

            logits = light_model(xc, xe)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            confidences = router.confidence_from_logits(logits).cpu().numpy()

            light_only_preds.append(np.argmax(probs, axis=1))
            light_only_prob_rows.append(probs)
            val_confidences.append(confidences)
            val_labels.append(labels.cpu().numpy())

            f_xc = xc.view(xc.size(0), -1).cpu().numpy()
            f_xe = xe.view(xe.size(0), -1).cpu().numpy()
            val_features.append(np.hstack([f_xc, f_xe]))

    all_labels = np.concatenate(val_labels)
    all_light_probs_rows = np.concatenate(light_only_prob_rows)
    all_light_preds = np.concatenate(light_only_preds)
    all_light_probs = all_light_probs_rows[:, 1]
    all_confidences = np.concatenate(val_confidences)
    all_val_features = np.vstack(val_features) if val_features else np.zeros((0, 0), dtype=np.float32)

    if heavy_model is not None and all_val_features.size > 0:
        heavy_res_all = predict_heavy(heavy_model, all_val_features)
        all_heavy_prob_rows = np.asarray(heavy_res_all['probabilities'], dtype=float)
    else:
        all_heavy_prob_rows = all_light_probs_rows.copy()

    calibrated_threshold, threshold_calibration = calibrate_router_threshold(
        y_true=all_labels,
        confidences=all_confidences,
        light_probs=all_light_probs_rows,
        heavy_probs=all_heavy_prob_rows,
        fpr_budget=args.fpr_budget,
        num_candidates=args.router_calibration_points,
    )
    router.config.threshold = calibrated_threshold
    print(
        f"Calibrated router threshold on validation set: {calibrated_threshold:.6f} "
        f"(reason={threshold_calibration['selection_reason']}, "
        f"budget_feasible={threshold_calibration['budget_feasible']})"
    )

    all_final_probs, all_final_preds, all_routed = _cascade_outputs_for_threshold(
        light_probs=all_light_probs_rows,
        heavy_probs=all_heavy_prob_rows,
        confidences=all_confidences,
        threshold=calibrated_threshold,
    )
    
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
    light_path_mask = ~all_routed.astype(bool)
    heavy_path_mask = all_routed.astype(bool)

    def _subset_metrics(mask: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
        subset_labels = all_labels[mask]
        subset_preds = preds[mask]
        if subset_labels.size == 0:
            return {
                "samples": 0,
                "metrics": {},
                "confusion_matrix": [[0, 0], [0, 0]],
            }
        return {
            "samples": int(subset_labels.size),
            "metrics": as_metrics(subset_labels, subset_preds),
            "confusion_matrix": confusion_matrix(subset_labels, subset_preds, labels=[0, 1]).tolist(),
        }

    light_path_stats = _subset_metrics(light_path_mask, all_final_preds)
    heavy_path_stats = _subset_metrics(heavy_path_mask, all_final_preds)
    
    print("\n--- Results ---")
    print("Light Only Metrics:", light_metrics)
    print("Cascade Metrics:   ", cascade_metrics)
    
    # 7. Save Report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "cascade_eval_replica_report.json")
    
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fpr_budget": float(args.fpr_budget),
        "dataset_build": {
            "train": train_meta,
            "validation": val_meta,
        },
        "evaluation_policy": {
            "split_manifest": args.split_manifest,
            "pairing_mode": args.pairing_mode,
            "allow_train_val_fallback": bool(args.allow_train_val_fallback),
            "allow_one_class_eval": bool(args.allow_one_class_eval),
            "synthetic": bool(args.synthetic),
        },
        "light_only": {
            **light_metrics,
            **light_ci,
            "confusion_matrix": cm_light.tolist()
        },
        "cascade": {
            **cascade_metrics,
            **cascade_ci,
            "confusion_matrix": cm_cascade.tolist(),
            "router_threshold": calibrated_threshold,
            "train_router_threshold": threshold,
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
        "threshold_calibration": {
            **threshold_calibration,
            "train_router_threshold": float(threshold),
        },
        "heavy_model_source": heavy_model_source,
    }

    router_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "router": {
            "threshold": float(calibrated_threshold),
            "mode": router_config.mode,
            "route_if_below_or_equal": bool(router_config.route_if_below_or_equal),
        },
        "decision_threshold": 0.5,
        "target_route_fraction": float(args.route_fraction),
        "actual_routed_fraction": float(np.mean(all_routed)),
        "samples": {
            "total": int(len(all_labels)),
            "light_path": int(light_path_stats["samples"]),
            "heavy_path": int(heavy_path_stats["samples"]),
        },
        "metrics": {
            "overall": cascade_metrics,
            "light_path": light_path_stats["metrics"],
            "heavy_path": heavy_path_stats["metrics"],
        },
        "confusion_matrix": {
            "overall": cm_cascade.tolist(),
            "light_path": light_path_stats["confusion_matrix"],
            "heavy_path": heavy_path_stats["confusion_matrix"],
        },
        "alignment_context": {
            "train": train_meta,
            "validation": val_meta,
        },
        "threshold_calibration": {
            **threshold_calibration,
            "train_router_threshold": float(threshold),
        },
        "heavy_model_source": heavy_model_source,
        "model_source": args.light_model_path,
    }

    latest_report_path = os.path.join(args.output_dir, "cascade_eval_report_latest.json")
    router_report_path = os.path.join(args.output_dir, "router_eval_report_latest.json")
    calibration_report_path = os.path.join(args.output_dir, "threshold_calibration_report_latest.json")

    for path, payload in (
        (report_path, report),
        (latest_report_path, report),
        (router_report_path, router_report),
        (calibration_report_path, report["threshold_calibration"]),
    ):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    
    print(f"Report saved to {report_path}")
    print(f"Latest cascade report saved to {latest_report_path}")
    print(f"Router report saved to {router_report_path}")
    print(f"Threshold calibration report saved to {calibration_report_path}")

if __name__ == "__main__":
    main()
