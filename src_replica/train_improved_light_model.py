import argparse
from collections import Counter
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

# Adjust path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from src_replica.hybrid_curriculum import training_pair_specs
from src_replica.data_resolvers import resolve_can_csv, resolve_eth_packet_csv
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bootstrap_ci(y_true, y_pred, n_resamples=1000, seed=42):
    """Return bootstrap 95% CI for F1 and accuracy using paired resampling."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {
            "f1_ci95": [0.0, 0.0],
            "accuracy_ci95": [0.0, 0.0],
        }

    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_vals, acc_vals = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1_vals.append(f1_score(yt, yp, zero_division=0))
        acc_vals.append(accuracy_score(yt, yp))

    return {
        "f1_ci95": [float(np.percentile(f1_vals, 2.5)), float(np.percentile(f1_vals, 97.5))],
        "accuracy_ci95": [float(np.percentile(acc_vals, 2.5)), float(np.percentile(acc_vals, 97.5))],
    }


def stratified_train_val_indices(labels, val_fraction=0.2, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(val_fraction * len(cls_idx))))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_kfold_indices(labels, n_splits=5, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    per_class_indices = {}
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        per_class_indices[cls] = np.array_split(idx, n_splits)

    folds = []
    for i in range(n_splits):
        val_parts = []
        train_parts = []
        for cls in per_class_indices:
            val_parts.append(per_class_indices[cls][i])
            train_parts.extend([per_class_indices[cls][j] for j in range(n_splits) if j != i])
        val_idx = np.concatenate(val_parts).tolist() if val_parts else []
        train_idx = np.concatenate(train_parts).tolist() if train_parts else []
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


def _extract_labels_for_concat(datasets_list):
    labels = []
    # Always extract labels at aligned-sample level so length matches ConcatDataset.
    for ds in datasets_list:
        for i in range(len(ds)):
            item = ds[i]
            if isinstance(item, dict):
                labels.append(int(item['label']))
            else:
                labels.append(int(item[1]))
    return np.asarray(labels, dtype=np.int64)


def _build_sample_groups(dataset_infos):
    groups = []
    pair_names = []
    for info in dataset_infos:
        groups.extend([info["group"]] * info["samples"])
        pair_names.extend([info["name"]] * info["samples"])
    return np.asarray(groups, dtype=object), np.asarray(pair_names, dtype=object)


def _inverse_frequency_map(values):
    counts = Counter(values)
    total = sum(counts.values())
    n_unique = max(len(counts), 1)
    return {key: float(total / (n_unique * count)) for key, count in counts.items() if count > 0}


def _make_subset_sampler(global_sample_weights, subset_indices):
    subset_indices = np.asarray(subset_indices, dtype=np.int64)
    subset_weights = np.asarray(global_sample_weights, dtype=np.float64)[subset_indices]
    weights_tensor = torch.as_tensor(subset_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(subset_indices), replacement=True)


def train_one_split(model, criterion, optimizer, scheduler, train_loader, val_loader, args, fold_tag="single"):
    best_f1 = 0.0
    best_state = None
    patience_ctr = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, dict):
                xc = batch['can'].to(args.device)
                xe = batch['eth'].to(args.device)
                labels = batch['label'].to(args.device)
            else:
                (xc, xe), labels = batch
                xc = xc.to(args.device)
                xe = xe.to(args.device)
                labels = labels.to(args.device)

            optimizer.zero_grad()
            logits = model(xc, xe)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    xc = batch['can'].to(args.device)
                    xe = batch['eth'].to(args.device)
                    labels = batch['label'].to(args.device)
                else:
                    (xc, xe), labels = batch
                    xc = xc.to(args.device)
                    xe = xe.to(args.device)
                    labels = labels.to(args.device)

                logits = model(xc, xe)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        f1 = f1_score(val_labels, val_preds, zero_division=0)
        acc = accuracy_score(val_labels, val_preds)
        scheduler.step(f1)

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[{fold_tag}] Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Val F1: {f1:.4f} | Acc: {acc:.4f}")

        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) >= args.min_epochs and patience_ctr >= args.patience:
            print(f"[{fold_tag}] Early stopping at epoch {epoch+1} (best epoch: {best_epoch}, best F1: {best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics from best checkpoint.
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                xc = batch['can'].to(args.device)
                xe = batch['eth'].to(args.device)
                labels = batch['label'].to(args.device)
            else:
                (xc, xe), labels = batch
                xc = xc.to(args.device)
                xe = xe.to(args.device)
                labels = labels.to(args.device)
            logits = model(xc, xe)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds = np.asarray(val_preds)
    val_labels = np.asarray(val_labels)
    ci = compute_bootstrap_ci(val_labels, val_preds, n_resamples=args.bootstrap_resamples, seed=args.seed)
    return {
        "f1": float(f1_score(val_labels, val_preds, zero_division=0)),
        "accuracy": float(accuracy_score(val_labels, val_preds)),
        "best_epoch": int(best_epoch),
        **ci,
    }

def train_light_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Training Improved Light Model on {device}")
    datasets_list = []
    dataset_infos = []
    
    # Define Features
    # Based on the engineered datasets
    can_features = STANDARD_CAN_FEATURES_16
    input_dim = len(can_features)
    print(f"Using {input_dim} CAN features.")

    # Load Data (Engineered)
    # We need to construct datasets manually since build_mixed_dataset in cascade_eval_replica 
    # points to raw data or is complex.
    
    # Data Mapping (Engineered CAN + NPY Images)
    # We will look for pairs in the engineered folder
    # Map of CAN file -> ETH NPY file (heuristic)
    # can_dos_train.csv -> eth_driving_01_injected_images-003.npy (Dos)
    # can_normal_train.csv -> eth_driving_01_original_images-006.npy (Normal)
    # can_fuzzy_train.csv -> eth_driving_02_injected_images-008.npy (Fuzzy)
    
    pairs = [
        # (Engineered CAN, ETH NPY, ETH CSV Base)
        # Attack pairs
        ("can_dos_train.csv", "eth_driving_01_injected_images-003.npy", "eth_driving_01_injected.csv"),
        ("can_fuzzy_train.csv", "eth_driving_02_injected_images-008.npy", "eth_driving_02_injected.csv"),
        ("can_gear_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
        ("can_rpm_train.csv", "eth_driving_02_original_images-005.npy", "eth_driving_02_original.csv"),
        # Normal pairs — critical for learning benign baseline
        ("can_normal_train.csv", "eth_driving_01_original_images-006.npy", "eth_driving_01_original.csv"),
        ("can_normal_train.csv", "eth_indoors_01_original_images.npy", "eth_indoors_01_original.csv"),
    ]
    pairs = [(spec.can_file, spec.eth_npy_file, spec) for spec in training_pair_specs()]
    
    # Track loaded pairs (CAN+ETH) to avoid exact duplicates
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
                    can_max_rows=args.max_rows # Will use full file if None
                )
                if len(ds) > 0:
                    datasets_list.append(ds)
                    dataset_infos.append({
                        "name": spec.name,
                        "group": spec.group,
                        "samples": len(ds),
                        "can_file": can_f,
                        "eth_npy_file": eth_n,
                    })
                    loaded_pairs.add(pair_key)
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

    print("Computing class and curriculum sampling weights from loaded training data...")
    all_labels = _extract_labels_for_concat(datasets_list)
    all_groups, all_pair_names = _build_sample_groups(dataset_infos)

    if len(all_groups) != len(all_labels) or len(all_pair_names) != len(all_labels):
        raise ValueError("Curriculum metadata length does not match aligned sample count.")

    split_keys = np.asarray(
        [f"{pair_name}:{int(label)}" for pair_name, label in zip(all_pair_names, all_labels)],
        dtype=object,
    )

    label_counts = Counter(int(label) for label in all_labels.tolist())
    group_counts = Counter(str(group) for group in all_groups.tolist())
    pair_counts = Counter(str(pair_name) for pair_name in all_pair_names.tolist())

    class_weight_map = _inverse_frequency_map([int(label) for label in all_labels.tolist()])
    group_weight_map = _inverse_frequency_map([str(group) for group in all_groups.tolist()])
    pair_weight_map = _inverse_frequency_map([str(pair_name) for pair_name in all_pair_names.tolist()])

    sample_weight_vector = np.asarray(
        [
            class_weight_map[int(label)] * pair_weight_map[str(pair_name)]
            for label, pair_name in zip(all_labels, all_pair_names)
        ],
        dtype=np.float64,
    )

    if len(all_labels) > 0:
        total = sum(label_counts.values())
        num_classes = max(label_counts.keys()) + 1
        class_weights = torch.zeros(num_classes, device=device)
        for cls, count in label_counts.items():
            class_weights[cls] = total / (num_classes * count)
        print(f"  Class distribution: {dict(label_counts)}")
        print(f"  Group distribution: {dict(group_counts)}")
        print(f"  Pair distribution: {dict(pair_counts)}")
        print(f"  Class weights: {class_weights.cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("  Warning: Could not compute class weights, using uniform.")
        class_weights = None

    full_ds = ConcatDataset(datasets_list)
    print(f"Total samples: {len(full_ds)}")

    if args.cv_folds > 1:
        print(f"Running stratified {args.cv_folds}-fold CV...")
        fold_reports = []
        folds = stratified_kfold_indices(split_keys, n_splits=args.cv_folds, seed=args.seed)
        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            model = TinyHybridStudent(input_dim=input_dim, hidden_dim=64, num_classes=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

            train_ds = Subset(full_ds, train_idx)
            val_ds = Subset(full_ds, val_idx)
            train_sampler = _make_subset_sampler(sample_weight_vector, train_idx)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

            fold_metrics = train_one_split(
                model, criterion, optimizer, scheduler,
                train_loader, val_loader, args, fold_tag=f"fold-{fold_id}"
            )
            fold_metrics["fold"] = fold_id
            fold_metrics["train_samples"] = int(len(train_idx))
            fold_metrics["val_samples"] = int(len(val_idx))
            fold_reports.append(fold_metrics)

            # Save per-fold checkpoints for reproducible reporting.
            fold_path = os.path.join(args.output_dir, f"student_tiny_improved_fold{fold_id}.pth")
            torch.save(model.state_dict(), fold_path)
            print(f"Saved fold {fold_id} model to {fold_path}")

        cv_summary = {
            "seed": int(args.seed),
            "cv_folds": int(args.cv_folds),
            "curriculum_pairs": dataset_infos,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "group_distribution": {str(k): int(v) for k, v in group_counts.items()},
            "pair_distribution": {str(k): int(v) for k, v in pair_counts.items()},
            "sample_weighting": {
                "strategy": "inverse_class_x_inverse_pair_frequency",
                "group_weights": {str(k): float(v) for k, v in group_weight_map.items()},
                "pair_weights": {str(k): float(v) for k, v in pair_weight_map.items()},
                "class_weights": {str(k): float(v) for k, v in class_weight_map.items()},
            },
            "folds": fold_reports,
            "f1_mean": float(np.mean([f["f1"] for f in fold_reports])),
            "f1_std": float(np.std([f["f1"] for f in fold_reports])),
            "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_reports])),
            "accuracy_std": float(np.std([f["accuracy"] for f in fold_reports])),
        }
        cv_path = os.path.join(args.output_dir, "student_tiny_improved_cv_report.json")
        with open(cv_path, "w", encoding="utf-8") as f:
            json.dump(cv_summary, f, indent=2)
        print(f"Saved CV report to {cv_path}")
    else:
        train_idx, val_idx = stratified_train_val_indices(split_keys, val_fraction=0.2, seed=args.seed)
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)

        model = TinyHybridStudent(input_dim=input_dim, hidden_dim=64, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

        train_sampler = _make_subset_sampler(sample_weight_vector, train_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        metrics = train_one_split(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader, args, fold_tag="single"
        )
        metrics.update({
            "seed": int(args.seed),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "curriculum_pairs": dataset_infos,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "group_distribution": {str(k): int(v) for k, v in group_counts.items()},
            "pair_distribution": {str(k): int(v) for k, v in pair_counts.items()},
            "sample_weighting": {
                "strategy": "inverse_class_x_inverse_pair_frequency",
                "group_weights": {str(k): float(v) for k, v in group_weight_map.items()},
                "pair_weights": {str(k): float(v) for k, v in pair_weight_map.items()},
                "class_weights": {str(k): float(v) for k, v in class_weight_map.items()},
            },
        })
        save_path = os.path.join(args.output_dir, "student_tiny_improved.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to {save_path}")

        metrics_path = os.path.join(args.output_dir, "student_tiny_improved_train_report.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved training report to {metrics_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=1, help="Set >1 for stratified k-fold training")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--min_epochs", type=int, default=5, help="Minimum epochs before early stopping")
    parser.add_argument("--bootstrap_resamples", type=int, default=1000, help="Bootstrap resamples for CI")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit rows for speed")
    parser.add_argument("--full_data", action="store_true", help="Use all pairs")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_light_model(args)

if __name__ == "__main__":
    main()
