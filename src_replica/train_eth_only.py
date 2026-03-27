"""
train_eth_only.py — Standalone ETH-only baseline training (Phase 2).

Trains a UnimodalETHModel on Ethernet packet data WITHOUT requiring any
paired CAN data. Automatically uses the Phase 1 DPI-enriched 64×64 images
when available (payload_entropy, payload_b0..b15, etc.), and gracefully
falls back to metadata-only images for legacy CSVs.

Usage:
    python src_replica/train_eth_only.py [options]

Key options:
    --data_dir      Directory containing replica_eth_smoke/ CSVs (default: datasets)
    --output_dir    Directory to save model and metrics (default: models)
    --epochs        Training epochs (default: 15)
    --cv_folds      Number of stratified CV folds; 1 = single split (default: 1)
    --batch_size    (default: 32)
    --window_size   Number of packets per ETH image window (default: 32)
    --overlap       Packet overlap between consecutive windows (default: 0)
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

from src_replica.architecture_unimodal import UnimodalETHModel
from src_replica.preprocessing_standard import (
    STANDARD_ETH_IMAGE_SIZE,
    build_eth_image_windows,
    standardize_eth_packet_dataframe,
    validate_eth_label_dataframe,
)

# ── ETH-only dataset ──────────────────────────────────────────────────────────

class ETHWindowDataset(Dataset):
    """
    Sliding-window ETH dataset derived from a **single** labeled packet CSV.

    Each item is ``(image_tensor, label)`` where:
      - ``image_tensor`` has shape ``(1, H, W)`` — single-channel 2D image
      - ``label`` is 0 (benign) or 1 (attack), taken as the max label in the window.

    Automatically benefits from Phase 1 DPI columns (payload_entropy, etc.)
    when present; zero-fills gracefully for legacy metadata-only CSVs.
    """

    def __init__(
        self,
        eth_csv_path: str,
        window_size: int = 32,
        overlap: int = 0,
        max_windows: Optional[int] = None,
    ):
        if not os.path.exists(eth_csv_path):
            raise FileNotFoundError(f"ETH CSV not found: {eth_csv_path}")

        raw_df = pd.read_csv(eth_csv_path)
        try:
            validate_eth_label_dataframe(raw_df, context=os.path.basename(eth_csv_path),
                                         require_label=True, require_provenance=True)
        except ValueError:
            # Fall back: accept CSVs without full provenance (old format)
            pass

        # Build images via the standard pipeline (DPI-aware)
        self.images = build_eth_image_windows(
            eth_csv_path,
            eth_window_size=window_size,
            eth_overlap=overlap,
            image_size=STANDARD_ETH_IMAGE_SIZE,
            max_windows=max_windows,
        )  # shape: (N, H, W)

        # Compute per-window labels from the CSV
        std_df = standardize_eth_packet_dataframe(raw_df)
        step = max(1, window_size - overlap)
        labels_arr = std_df["Label"].astype(int).to_numpy() if "Label" in std_df.columns \
            else np.zeros(len(std_df), dtype=int)

        n = len(self.images)
        self._labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            s = i * step
            e = min(s + window_size, len(labels_arr))
            self._labels[i] = int(labels_arr[s:e].max()) if e > s else 0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        return img, label

    @property
    def window_labels(self) -> np.ndarray:
        return self._labels.copy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_eth_csvs(data_dir: str) -> List[str]:
    """Find all labeled ETH packet CSVs in replica_eth_smoke/."""
    smoke_dir = os.path.join(data_dir, "replica_eth_smoke")
    paths = []
    if os.path.isdir(smoke_dir):
        for name in sorted(os.listdir(smoke_dir)):
            if name.startswith("eth_") and name.endswith("_replica_packets.csv"):
                paths.append(os.path.join(smoke_dir, name))
    return paths


def _stratified_split(labels: np.ndarray, val_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx); rng.shuffle(val_idx)
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
    wts = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items() if cnt > 0}
    sample_weights = torch.tensor([wts[int(l)] for l in labels[subset_indices]], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(subset_indices), replacement=True)


def _train_one_split(model, criterion, optimizer, scheduler, train_loader, val_loader, args, tag="single"):
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
                preds.extend(torch.argmax(model(batch_x.to(args.device)), 1).cpu().numpy())
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
            print(f"  [{tag}] Early stop at epoch {epoch+1} (best F1: {best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds.extend(torch.argmax(model(batch_x.to(args.device)), 1).cpu().numpy())
            trues.extend(batch_y.numpy())

    return {
        "f1": float(f1_score(trues, preds, zero_division=0)),
        "accuracy": float(accuracy_score(trues, preds)),
        "best_epoch": best_epoch,
    }


# ── Main training entry-point ──────────────────────────────────────────────────

def train_eth_only(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"[ETH-only] Training on {device}")
    print(f"  ETH image size: {STANDARD_ETH_IMAGE_SIZE}×{STANDARD_ETH_IMAGE_SIZE} (Phase 1 DPI-enriched)")

    csv_paths = _discover_eth_csvs(args.data_dir)
    if not csv_paths:
        print("No ETH packet CSVs found in datasets/replica_eth_smoke/.")
        print("Run: python setup_datasets.py --skip-download")
        return

    datasets, names = [], []
    for path in csv_paths:
        try:
            ds = ETHWindowDataset(path, window_size=args.window_size, overlap=args.overlap)
            if len(ds) > 0:
                datasets.append(ds)
                names.append(os.path.basename(path))
                print(f"  Loaded {os.path.basename(path)}: {len(ds)} windows")
        except Exception as exc:
            print(f"  [warn] Could not load {os.path.basename(path)}: {exc}")

    if not datasets:
        print("No ETH windows loaded. Exiting.")
        return

    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset(datasets)
    all_labels = np.concatenate([ds.window_labels for ds in datasets])
    label_counts = Counter(int(l) for l in all_labels)
    print(f"  Total windows: {len(all_labels)} | Class distribution: {dict(label_counts)}")

    num_classes = 2
    class_wts = torch.zeros(num_classes, device=device)
    total = sum(label_counts.values())
    for cls, cnt in label_counts.items():
        class_wts[cls] = total / (num_classes * cnt)
    criterion = nn.CrossEntropyLoss(weight=class_wts)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.cv_folds > 1:
        print(f"  Running {args.cv_folds}-fold stratified CV...")
        folds = _stratified_kfold(all_labels, n_splits=args.cv_folds, seed=args.seed)
        fold_reports = []
        for fold_id, (train_idx, val_idx) in enumerate(folds, 1):
            model = UnimodalETHModel(num_classes=num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

            train_sampler = _make_weighted_sampler(all_labels, np.array(train_idx))
            train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, sampler=train_sampler)
            val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False)

            metrics = _train_one_split(model, criterion, optimizer, scheduler, train_loader, val_loader, args, f"fold-{fold_id}")
            metrics["fold"] = fold_id
            fold_reports.append(metrics)

            fold_path = os.path.join(args.output_dir, f"eth_only_model_fold{fold_id}.pth")
            torch.save(model.state_dict(), fold_path)
            print(f"  Saved fold {fold_id} model → {fold_path}")

        summary = {
            "modality": "ETH-only",
            "model": "UnimodalETHModel",
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "window_size": args.window_size,
            "image_size": STANDARD_ETH_IMAGE_SIZE,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "folds": fold_reports,
            "f1_mean": float(np.mean([f["f1"] for f in fold_reports])),
            "f1_std": float(np.std([f["f1"] for f in fold_reports])),
            "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_reports])),
        }
        report_path = os.path.join(args.output_dir, "eth_only_cv_report.json")
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"  Saved CV report → {report_path}")
        print(f"  ETH-only F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")

    else:
        train_idx, val_idx = _stratified_split(all_labels, val_frac=0.2, seed=args.seed)
        model = UnimodalETHModel(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        train_sampler = _make_weighted_sampler(all_labels, np.array(train_idx))
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False)

        metrics = _train_one_split(model, criterion, optimizer, scheduler, train_loader, val_loader, args)
        metrics.update({
            "modality": "ETH-only",
            "model": "UnimodalETHModel",
            "seed": args.seed,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "window_size": args.window_size,
            "image_size": STANDARD_ETH_IMAGE_SIZE,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        })

        save_path = os.path.join(args.output_dir, "eth_only_model.pth")
        torch.save(model.state_dict(), save_path)
        report_path = os.path.join(args.output_dir, "eth_only_train_report.json")
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"  Saved model → {save_path}")
        print(f"  Saved report → {report_path}")
        print(f"  ETH-only F1: {metrics['f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train ETH-only unimodal IDS baseline (Phase 2).")
    p.add_argument("--data_dir", default="datasets")
    p.add_argument("--output_dir", default="models")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min_epochs", type=int, default=5)
    args = p.parse_args()
    train_eth_only(args)


if __name__ == "__main__":
    main()
