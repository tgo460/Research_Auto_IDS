"""
train_can_only.py — Standalone CAN-only baseline training (Phase 2).

Trains a UnimodalCANModel on CAN bus data WITHOUT requiring any paired
Ethernet data. This establishes the single-bus CAN baseline that the
multimodal model must surpass.

Usage:
    python src_replica/train_can_only.py [options]

Key options:
    --data_dir      Directory that contains CAN CSV files (default: datasets)
    --output_dir    Directory to save model and metrics (default: models)
    --epochs        Training epochs (default: 15)
    --cv_folds      Number of stratified CV folds; 1 = single split (default: 1)
    --batch_size    (default: 64)
    --window_size   CAN sliding-window size in frames (default: 100)
    --overlap       Overlap between consecutive windows (default: 50)
    --lr            Learning-rate (default: 0.001)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.architecture_unimodal import UnimodalCANModel
from src_replica.features_can_replica import add_can_engineered_features
from src_replica.preprocessing_standard import (
    STANDARD_CAN_FEATURES_16,
    standardize_can_dataframe,
)

# ── CAN-only dataset ──────────────────────────────────────────────────────────

class CANWindowDataset(Dataset):
    """
    Sliding-window CAN dataset.

    Each item is ``(window_tensor, label)`` where:
      - ``window_tensor`` has shape ``(window_size, n_features)``
      - ``label`` is 0 (normal) or 1 (attack), taken as the max label in
        the window (any attack frame contaminates the whole window).
    """

    def __init__(
        self,
        can_csv_path: str,
        can_features: List[str],
        window_size: int = 100,
        overlap: int = 50,
        max_rows: Optional[int] = None,
    ):
        if not os.path.exists(can_csv_path):
            raise FileNotFoundError(f"CAN CSV not found: {can_csv_path}")

        df = pd.read_csv(can_csv_path)
        if max_rows is not None:
            df = df.head(max_rows)

        # Standardise + engineer
        df = standardize_can_dataframe(df)
        raw_feats = {"CAN_ID", "DLC", "Label", "Timestamp"} | {f"D{i}" for i in range(8)}
        if any(f not in raw_feats for f in can_features):
            df = add_can_engineered_features(df)

        missing = [f for f in can_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing CAN features after engineering: {missing}")

        self.values = df[can_features].to_numpy(dtype=np.float32)
        self.labels = df["Label"].astype(int).to_numpy()
        self.window_size = window_size
        self.step = max(1, window_size - overlap)

        n_windows = max(0, (len(self.values) - window_size) // self.step + 1) \
                    if len(self.values) >= window_size else 0
        self._n = n_windows

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.step
        e = s + self.window_size
        window = self.values[s:e]
        label = int(self.labels[s:e].max())
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    @property
    def window_labels(self) -> np.ndarray:
        """Array of labels for every window (used for stratified splitting)."""
        return np.array([int(self.labels[i * self.step: i * self.step + self.window_size].max())
                         for i in range(self._n)], dtype=np.int64)


# ── Helpers ───────────────────────────────────────────────────────────────────

_CAN_FILES = [
    "can_dos_train.csv",
    "can_fuzzy_train.csv",
    "can_gear_train.csv",
    "can_rpm_train.csv",
    "can_normal_train.csv",
]


def _load_can_datasets(data_dir: str, can_features: List[str], window_size: int, overlap: int,
                       max_rows: Optional[int]) -> Tuple[List[CANWindowDataset], List[str]]:
    datasets, names = [], []
    for fname in _CAN_FILES:
        # Try raw first, then engineered sub-dir
        for sub in ["", os.path.join("replica_can_b1_engineered", "")]:
            path = os.path.join(data_dir, sub, fname)
            if os.path.exists(path):
                try:
                    ds = CANWindowDataset(path, can_features, window_size, overlap, max_rows)
                    if len(ds) > 0:
                        datasets.append(ds)
                        names.append(fname)
                        print(f"  Loaded {fname}: {len(ds)} windows")
                except Exception as exc:
                    print(f"  [warn] Could not load {fname}: {exc}")
                break
    return datasets, names


def _stratified_split(labels: np.ndarray, val_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _stratified_kfold(labels: np.ndarray, n_splits: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    per_class = {}
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        per_class[cls] = np.array_split(idx, n_splits)
    folds = []
    for i in range(n_splits):
        val_parts = [per_class[c][i] for c in per_class]
        train_parts = [per_class[c][j] for c in per_class for j in range(n_splits) if j != i]
        val_idx = np.concatenate(val_parts).tolist()
        train_idx = np.concatenate(train_parts).tolist()
        rng.shuffle(train_idx); rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


def _make_weighted_sampler(labels: np.ndarray, subset_indices: np.ndarray) -> WeightedRandomSampler:
    counts = Counter(int(l) for l in labels[subset_indices])
    total = len(subset_indices)
    class_weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items() if cnt > 0}
    sample_weights = torch.tensor([class_weights[int(l)] for l in labels[subset_indices]], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(subset_indices), replacement=True)


def _train_one_split(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    tag: str = "single",
) -> dict:
    best_f1, best_state, patience_ctr, best_epoch = 0.0, None, 0, 0

    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x.to(args.device))
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(batch_y.numpy())

        f1 = f1_score(trues, preds, zero_division=0)
        acc = accuracy_score(trues, preds)
        scheduler.step(f1)
        print(f"  [{tag}] Epoch {epoch+1}/{args.epochs} | Val F1: {f1:.4f} | Acc: {acc:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, patience_ctr = f1, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if (epoch + 1) >= args.min_epochs and patience_ctr >= args.patience:
            print(f"  [{tag}] Early stop at epoch {epoch+1} (best epoch: {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval on best checkpoint
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x.to(args.device))
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(batch_y.numpy())

    return {
        "f1": float(f1_score(trues, preds, zero_division=0)),
        "accuracy": float(accuracy_score(trues, preds)),
        "best_epoch": best_epoch,
    }


# ── Main training entry-point ──────────────────────────────────────────────────

def train_can_only(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"[CAN-only] Training on {device}")

    can_features = STANDARD_CAN_FEATURES_16
    print(f"  CAN features ({len(can_features)}): {can_features}")

    datasets, names = _load_can_datasets(
        args.data_dir, can_features, args.window_size, args.overlap, args.max_rows
    )
    if not datasets:
        print("No CAN datasets found. Download the Car-Hacking Dataset first.")
        print("  Expected files in datasets/: can_dos_train.csv, can_fuzzy_train.csv, etc.")
        return

    # Concatenate all window labels for stratified splitting
    all_labels = np.concatenate([ds.window_labels for ds in datasets])
    offsets = np.cumsum([0] + [len(ds) for ds in datasets[:-1]])

    # Build flat index arrays
    flat_indices = np.arange(sum(len(ds) for ds in datasets))
    label_counts = Counter(int(l) for l in all_labels)
    print(f"  Total windows: {len(flat_indices)} | Class distribution: {dict(label_counts)}")

    # Compute class weights for loss
    num_classes = 2
    class_wts = torch.zeros(num_classes, device=device)
    total = sum(label_counts.values())
    for cls, cnt in label_counts.items():
        class_wts[cls] = total / (num_classes * cnt)
    criterion = nn.CrossEntropyLoss(weight=class_wts)

    # Flat dataset (ConcatDataset-like)
    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset(datasets)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.cv_folds > 1:
        print(f"  Running {args.cv_folds}-fold stratified CV...")
        folds = _stratified_kfold(all_labels, n_splits=args.cv_folds, seed=args.seed)
        fold_reports = []
        for fold_id, (train_idx, val_idx) in enumerate(folds, 1):
            model = UnimodalCANModel(input_dim=len(can_features), num_classes=num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

            train_sampler = _make_weighted_sampler(all_labels, np.array(train_idx))
            train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, sampler=train_sampler)
            val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False)

            metrics = _train_one_split(model, criterion, optimizer, scheduler, train_loader, val_loader, args, f"fold-{fold_id}")
            metrics["fold"] = fold_id
            fold_reports.append(metrics)

            fold_path = os.path.join(args.output_dir, f"can_only_model_fold{fold_id}.pth")
            torch.save(model.state_dict(), fold_path)
            print(f"  Saved fold {fold_id} model → {fold_path}")

        summary = {
            "modality": "CAN-only",
            "model": "UnimodalCANModel",
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "can_features": can_features,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "folds": fold_reports,
            "f1_mean": float(np.mean([f["f1"] for f in fold_reports])),
            "f1_std": float(np.std([f["f1"] for f in fold_reports])),
            "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_reports])),
        }
        report_path = os.path.join(args.output_dir, "can_only_cv_report.json")
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"  Saved CV report → {report_path}")
        print(f"  CAN-only F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")

    else:
        train_idx, val_idx = _stratified_split(all_labels, val_frac=0.2, seed=args.seed)
        model = UnimodalCANModel(input_dim=len(can_features), num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        train_sampler = _make_weighted_sampler(all_labels, np.array(train_idx))
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False)

        metrics = _train_one_split(model, criterion, optimizer, scheduler, train_loader, val_loader, args)
        metrics.update({
            "modality": "CAN-only",
            "model": "UnimodalCANModel",
            "seed": args.seed,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "window_size": args.window_size,
            "overlap": args.overlap,
            "can_features": can_features,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        })

        save_path = os.path.join(args.output_dir, "can_only_model.pth")
        torch.save(model.state_dict(), save_path)
        report_path = os.path.join(args.output_dir, "can_only_train_report.json")
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"  Saved model → {save_path}")
        print(f"  Saved report → {report_path}")
        print(f"  CAN-only F1: {metrics['f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train CAN-only unimodal IDS baseline (Phase 2).")
    p.add_argument("--data_dir", default="datasets")
    p.add_argument("--output_dir", default="models")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--cv_folds", type=int, default=1, help="1 = single train/val split")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min_epochs", type=int, default=5)
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()
    train_can_only(args)


if __name__ == "__main__":
    main()
