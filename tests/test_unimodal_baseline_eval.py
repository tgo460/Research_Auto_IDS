import torch
from torch.utils.data import Dataset

from src_replica.unimodal_baseline_eval_replica import _extract_features_and_labels


class _DummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        scale = float(idx + 1)
        return {
            "can": torch.full((2, 16), scale, dtype=torch.float32),
            "eth": torch.full((1, 1, 4, 4), scale * 2.0, dtype=torch.float32),
            "label": torch.tensor(idx % 2, dtype=torch.long),
        }


def test_extract_features_and_labels_can_only_shape_and_labels():
    X, y = _extract_features_and_labels(_DummyDataset(), mode="can_only")
    assert X.shape == (2, 32)
    assert y.tolist() == [0, 1]
    assert float(X[1].sum()) > float(X[0].sum())


def test_extract_features_and_labels_eth_only_shape_and_labels():
    X, y = _extract_features_and_labels(_DummyDataset(), mode="eth_only")
    assert X.shape == (2, 16)
    assert y.tolist() == [0, 1]
    assert float(X[1].sum()) > float(X[0].sum())
