"""
tests/test_unimodal_training.py

Phase 2 – unit tests for standalone unimodal training baselines.

Tests verify:
 1. CANWindowDataset yields (window_tensor, label) with correct shapes.
 2. ETHWindowDataset yields (image_tensor, label) with correct shapes.
 3. UnimodalCANModel forward pass: (B, L, C) → (B, 2) logits.
 4. UnimodalETHModel forward pass: (B, 1, H, W) → (B, 2) logits.
 5. UnimodalETHModel handles batched 5-D input (B, T, 1, H, W).
 6. UnimodalCANModel.extract_features returns intermediate features.
 7. UnimodalETHModel.extract_features returns intermediate features.
 8. A short training loop for CAN-only completes without error on synthetic data.
 9. A short training loop for ETH-only completes without error on synthetic data.
10. CANWindowDataset gracefully raises when file is missing.
11. ETHWindowDataset gracefully raises when file is missing.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src_replica.architecture_unimodal import UnimodalCANModel, UnimodalETHModel
from src_replica.preprocessing_standard import STANDARD_ETH_IMAGE_SIZE
from src_replica.train_can_only import CANWindowDataset
from src_replica.train_eth_only import ETHWindowDataset


# ── Synthetic data fixtures ───────────────────────────────────────────────────

def _make_can_csv(tmp_path, n_rows: int = 300, attack_frac: float = 0.3) -> str:
    """Create a minimal CAN CSV (no DLC column normalisation issues)."""
    rng = np.random.default_rng(0)
    n_attack = int(n_rows * attack_frac)
    n_normal = n_rows - n_attack
    labels = np.array([0] * n_normal + [1] * n_attack)
    rng.shuffle(labels)

    path = tmp_path / "can_synthetic.csv"
    df = pd.DataFrame({
        "Timestamp": np.linspace(0.0, 3.0, n_rows),
        "CAN_ID": rng.uniform(0.0, 1.0, n_rows),
        "DLC": np.full(n_rows, 1.0),
        **{f"D{i}": rng.uniform(0.0, 1.0, n_rows) for i in range(8)},
        "Label": labels,
    })
    df.to_csv(path, index=False)
    return str(path)


def _make_eth_csv(tmp_path, n_rows: int = 120, attack_frac: float = 0.4) -> str:
    """Create a labeled ETH packet CSV including Phase 1 DPI columns."""
    rng = np.random.default_rng(1)
    n_attack = int(n_rows * attack_frac)
    n_normal = n_rows - n_attack
    labels = np.array([0] * n_normal + [1] * n_attack)
    rng.shuffle(labels)

    path = tmp_path / "eth_synthetic_replica_packets.csv"
    df = pd.DataFrame({
        "timestamp_sec": np.arange(n_rows),
        "timestamp_usec": np.zeros(n_rows, dtype=int),
        "captured_len": rng.integers(64, 512, n_rows),
        "original_len": rng.integers(64, 512, n_rows),
        "Label": labels,
        "session_id": ["s1"] * n_rows,
        "attack_type": ["benign" if l == 0 else "avtp_injection" for l in labels],
        "label_source": ["packet_ground_truth"] * n_rows,
        "label_granularity": ["packet"] * n_rows,
        # Phase 1 DPI columns
        "eth_type": np.full(n_rows, 0x0800 / 65535.0),
        "ip_proto": np.full(n_rows, 17.0 / 255.0),
        "src_port": rng.uniform(0, 1, n_rows),
        "dst_port": rng.uniform(0, 1, n_rows),
        "payload_len": rng.integers(0, 100, n_rows),
        "payload_entropy": rng.uniform(0.0, 1.0, n_rows),
        **{f"payload_b{i}": rng.uniform(0.0, 1.0, n_rows) for i in range(16)},
    })
    df.to_csv(path, index=False)
    return str(path)


# ── Dataset shape tests ───────────────────────────────────────────────────────

class TestCANWindowDataset:
    def test_yields_correct_window_shape(self, tmp_path):
        path = _make_can_csv(tmp_path, n_rows=300)
        ds = CANWindowDataset(
            path,
            can_features=["CAN_ID", "DLC"] + [f"D{i}" for i in range(8)],
            window_size=50,
            overlap=25,
        )
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (50, 10), f"Expected (50, 10), got {x.shape}"
        assert y.dtype == torch.int64

    def test_uses_standard_can_features_16(self, tmp_path):
        from src_replica.preprocessing_standard import STANDARD_CAN_FEATURES_16
        path = _make_can_csv(tmp_path, n_rows=300)
        ds = CANWindowDataset(
            path,
            can_features=STANDARD_CAN_FEATURES_16,
            window_size=100,
            overlap=50,
        )
        x, _ = ds[0]
        assert x.shape == (100, 16), f"Expected (100, 16), got {x.shape}"

    def test_window_labels_have_correct_length(self, tmp_path):
        path = _make_can_csv(tmp_path, n_rows=300)
        ds = CANWindowDataset(path, can_features=["CAN_ID", "DLC"] + [f"D{i}" for i in range(8)],
                              window_size=50, overlap=0)
        assert len(ds.window_labels) == len(ds)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CANWindowDataset(
                str(tmp_path / "nonexistent.csv"),
                can_features=["CAN_ID"],
                window_size=10,
                overlap=0,
            )


class TestETHWindowDataset:
    def test_yields_correct_image_shape(self, tmp_path):
        path = _make_eth_csv(tmp_path, n_rows=120)
        ds = ETHWindowDataset(path, window_size=10, overlap=0)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (1, STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE), \
            f"Expected (1, {STANDARD_ETH_IMAGE_SIZE}, {STANDARD_ETH_IMAGE_SIZE}), got {x.shape}"
        assert y.dtype == torch.int64

    def test_window_labels_have_correct_length(self, tmp_path):
        path = _make_eth_csv(tmp_path, n_rows=80)
        ds = ETHWindowDataset(path, window_size=8, overlap=0)
        assert len(ds.window_labels) == len(ds)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ETHWindowDataset(str(tmp_path / "nonexistent.csv"))

    def test_image_values_in_unit_range(self, tmp_path):
        path = _make_eth_csv(tmp_path, n_rows=60)
        ds = ETHWindowDataset(path, window_size=5, overlap=0)
        x, _ = ds[0]
        assert float(x.min()) >= 0.0
        assert float(x.max()) <= 1.0


# ── Model architecture tests ──────────────────────────────────────────────────

class TestUnimodalCANModel:
    def test_forward_pass_shape(self):
        model = UnimodalCANModel(input_dim=16, num_classes=2)
        x = torch.randn(4, 100, 16)
        logits = model(x)
        assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"

    def test_forward_pass_different_sequence_lengths(self):
        model = UnimodalCANModel(input_dim=10)
        for seq_len in [50, 100, 200]:
            logits = model(torch.randn(2, seq_len, 10))
            assert logits.shape == (2, 2)

    def test_extract_features_shape(self):
        model = UnimodalCANModel(input_dim=16, hidden_dim=64)
        x = torch.randn(3, 100, 16)
        feats = model.extract_features(x)
        assert feats.shape == (3, 64), f"Expected (3, 64), got {feats.shape}"

    def test_parameter_count_is_reasonable(self):
        model = UnimodalCANModel(input_dim=16)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params < 500_000, f"Model too large: {n_params} parameters"

    def test_output_is_finite(self):
        model = UnimodalCANModel()
        out = model(torch.randn(2, 100, 16))
        assert torch.all(torch.isfinite(out))


class TestUnimodalETHModel:
    def test_forward_pass_4d_shape(self):
        model = UnimodalETHModel(num_classes=2)
        x = torch.randn(4, 1, STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE)
        logits = model(x)
        assert logits.shape == (4, 2)

    def test_forward_pass_5d_temporal(self):
        model = UnimodalETHModel(num_classes=2)
        # (B, T, 1, H, W)
        x = torch.randn(2, 3, 1, STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE)
        logits = model(x)
        assert logits.shape == (2, 2)

    def test_extract_features_shape(self):
        model = UnimodalETHModel(feat_dim=32)
        x = torch.randn(3, 1, STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE)
        feats = model.extract_features(x)
        assert feats.shape == (3, 32)

    def test_output_is_finite(self):
        model = UnimodalETHModel()
        out = model(torch.randn(2, 1, STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE))
        assert torch.all(torch.isfinite(out))

    def test_works_with_legacy_32x32_images(self):
        """Model must still work on 32×32 images (AdaptiveAvgPool2d is size-agnostic)."""
        model = UnimodalETHModel()
        out = model(torch.randn(2, 1, 32, 32))
        assert out.shape == (2, 2)


# ── Mini training loop tests ──────────────────────────────────────────────────

def _make_simple_args(device="cpu"):
    return SimpleNamespace(
        epochs=3,
        min_epochs=1,
        patience=5,
        device=torch.device(device),
    )


class TestCANTrainingLoop:
    def test_short_loop_runs_without_error(self, tmp_path):
        path = _make_can_csv(tmp_path, n_rows=300)
        can_features = ["CAN_ID", "DLC"] + [f"D{i}" for i in range(8)]
        ds = CANWindowDataset(path, can_features=can_features, window_size=50, overlap=0)

        loader = DataLoader(ds, batch_size=16, shuffle=True)
        model = UnimodalCANModel(input_dim=10)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        from src_replica.train_can_only import _train_one_split
        metrics = _train_one_split(
            model, criterion, optimizer,
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
            loader, loader,
            _make_simple_args(),
            tag="can-test",
        )
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_dataset_label_distribution(self, tmp_path):
        path = _make_can_csv(tmp_path, n_rows=300, attack_frac=0.3)
        feat_cols = ["CAN_ID", "DLC"] + [f"D{i}" for i in range(8)]
        ds = CANWindowDataset(path, can_features=feat_cols, window_size=10, overlap=0)
        labels = ds.window_labels
        assert (labels == 0).any(), "Expected some normal windows"
        assert (labels == 1).any(), "Expected some attack windows"


class TestETHTrainingLoop:
    def test_short_loop_runs_without_error(self, tmp_path):
        path = _make_eth_csv(tmp_path, n_rows=120)
        ds = ETHWindowDataset(path, window_size=10, overlap=0)

        loader = DataLoader(ds, batch_size=8, shuffle=True)
        model = UnimodalETHModel()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        from src_replica.train_eth_only import _train_one_split
        metrics = _train_one_split(
            model, criterion, optimizer,
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
            loader, loader,
            _make_simple_args(),
            tag="eth-test",
        )
        assert "f1" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_dataset_label_distribution(self, tmp_path):
        path = _make_eth_csv(tmp_path, n_rows=80, attack_frac=0.4)
        ds = ETHWindowDataset(path, window_size=5, overlap=0)
        labels = ds.window_labels
        assert (labels == 0).any(), "Expected some benign windows"
        assert (labels == 1).any(), "Expected some attack windows"
