import torch
from torch.utils.data import Dataset

from src_replica.ablation_eval_replica import ModalityAblationDataset, _metrics


class _DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "can": torch.ones(2, 16, dtype=torch.float32),
            "eth": torch.ones(1, 1, 32, 32, dtype=torch.float32),
            "label": torch.tensor(1, dtype=torch.long),
        }


def test_modality_ablation_dataset_zeroes_eth_for_can_only():
    ds = ModalityAblationDataset(_DummyDataset(), mode="can_only")
    sample = ds[0]
    assert float(sample["can"].sum()) > 0.0
    assert float(sample["eth"].sum()) == 0.0


def test_modality_ablation_dataset_zeroes_can_for_eth_only():
    ds = ModalityAblationDataset(_DummyDataset(), mode="eth_only")
    sample = ds[0]
    assert float(sample["can"].sum()) == 0.0
    assert float(sample["eth"].sum()) > 0.0


def test_modality_ablation_dataset_preserves_both_for_fused():
    ds = ModalityAblationDataset(_DummyDataset(), mode="fused")
    sample = ds[0]
    assert float(sample["can"].sum()) > 0.0
    assert float(sample["eth"].sum()) > 0.0


def test_metrics_records_single_class_label_summary():
    metrics = _metrics([1, 1, 1], [1, 1, 1])
    assert metrics["positives"] == 3
    assert metrics["negatives"] == 0
    assert metrics["classes_present"] == [1]
    assert metrics["is_single_class"] is True
