import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src_replica.correlation_replica import correlate_can_eth
from src_replica.features_can_replica import add_can_engineered_features
from src_replica.preprocessing_standard import (
    STANDARD_ETH_IMAGE_SIZE,
    STANDARD_ETH_REPRESENTATION,
    build_eth_image_windows,
    standardize_can_dataframe,
)


def _candidate_can_timestamp_csvs(can_csv_path: str) -> List[str]:
    can_dir = os.path.dirname(can_csv_path)
    fname = os.path.basename(can_csv_path)
    if os.path.basename(can_dir) == "replica_can_b1_engineered":
        return [can_csv_path]
    return [os.path.join(can_dir, "replica_can_b1_engineered", fname)]


def _load_can_timestamps(
    can_csv_path: str,
    n_rows: int,
    row_start: int = 0,
    row_stop: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    for candidate in _candidate_can_timestamp_csvs(can_csv_path):
        if not os.path.exists(candidate):
            continue
        try:
            donor_df = pd.read_csv(candidate, usecols=["Timestamp"])
        except Exception:
            continue
        start = max(int(row_start), 0)
        stop = len(donor_df) if row_stop is None else int(row_stop)
        donor_slice = donor_df.iloc[start:stop]
        if len(donor_slice) < n_rows:
            continue
        return donor_slice["Timestamp"].to_numpy(dtype=np.float64)[:n_rows], candidate

    synthetic = np.arange(n_rows, dtype=np.float64) * 0.001
    return synthetic, "synthetic_sequence"


def _candidate_eth_label_csvs(
    eth_packet_csv_path: str,
    eth_npy_path: str,
    eth_label_csv_path: Optional[str] = None,
) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def add(path: Optional[str]) -> None:
        if not path:
            return
        norm = os.path.normpath(path)
        if norm in seen:
            return
        seen.add(norm)
        candidates.append(path)

    add(eth_label_csv_path)
    add(eth_packet_csv_path)

    packet_dir = os.path.dirname(eth_packet_csv_path)
    datasets_dir = (
        os.path.dirname(packet_dir)
        if os.path.basename(packet_dir) == "replica_eth_smoke"
        else packet_dir
    )

    packet_name = os.path.basename(eth_packet_csv_path)
    if packet_name.endswith("_replica_packets.csv"):
        base = packet_name[:-len("_replica_packets.csv")]
        add(os.path.join(datasets_dir, f"{base}.csv"))
        add(os.path.join(datasets_dir, f"{base}_preprocessed.csv"))

    npy_name = os.path.basename(eth_npy_path)
    npy_base = npy_name.split("_images")[0] if "_images" in npy_name else os.path.splitext(npy_name)[0]
    add(os.path.join(packet_dir, f"{npy_base}_replica_packets.csv"))
    add(os.path.join(datasets_dir, f"{npy_base}_preprocessed.csv"))
    add(os.path.join(datasets_dir, f"{npy_base}.csv"))

    return [path for path in candidates if os.path.exists(path)]


def _load_eth_labels(
    eth_packet_csv_path: str,
    eth_npy_path: str,
    eth_label_csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    searched = _candidate_eth_label_csvs(
        eth_packet_csv_path=eth_packet_csv_path,
        eth_npy_path=eth_npy_path,
        eth_label_csv_path=eth_label_csv_path,
    )
    for candidate in _candidate_eth_label_csvs(
        eth_packet_csv_path=eth_packet_csv_path,
        eth_npy_path=eth_npy_path,
        eth_label_csv_path=eth_label_csv_path,
    ):
        try:
            eth_df = pd.read_csv(candidate)
        except Exception:
            continue
        if "Label" not in eth_df.columns:
            continue
        labels = pd.to_numeric(eth_df["Label"], errors="coerce").fillna(0).astype(int).to_numpy()
        if labels.size > 0:
            return labels, candidate

    searched_display = ", ".join(os.path.basename(path) for path in searched) if searched else "(no candidate files found)"
    raise ValueError(
        "ETH labels are required for supervised loading; no candidate ETH label CSV contained a "
        f"'Label' column. Searched: {searched_display}"
    )

class CorrelatedHybridVehicleDataset(Dataset):
    """
    Replica dataset that aligns CAN and ETH windows via timestamp/session correlation (C1).
    """
    def __init__(self, 
                 can_csv_path: str,
                 eth_packet_csv_path: str,
                 eth_npy_path: Optional[str],
                 can_features: List[str],
                 can_window_size: int = 100,
                 can_overlap: int = 50,
                 eth_window_size: int = 50,
                 eth_overlap: int = 25,
                 tolerance_ms: float = 100.0,
                 time_mode: str = 'relative_session',
                 can_max_rows: Optional[int] = None,
                 can_row_start: Optional[int] = None,
                 can_row_stop: Optional[int] = None,
                 eth_max_frames: Optional[int] = None,
                 label_policy: str = 'max',
                 eth_label_csv_path: Optional[str] = None,
                 eth_representation: str = STANDARD_ETH_REPRESENTATION,
                 eth_image_size: int = STANDARD_ETH_IMAGE_SIZE):
        
        if not os.path.exists(can_csv_path):
            raise FileNotFoundError(f"CAN CSV not found: {can_csv_path}")
        if not os.path.exists(eth_packet_csv_path):
            raise FileNotFoundError(f"ETH packet CSV not found: {eth_packet_csv_path}")
        if eth_representation != STANDARD_ETH_REPRESENTATION and eth_npy_path and not os.path.exists(eth_npy_path):
            raise FileNotFoundError(f"ETH image NPY not found: {eth_npy_path}")

        self.can_features = can_features
        self.can_window_size = can_window_size
        self.can_overlap = can_overlap
        self.eth_window_size = eth_window_size
        self.eth_overlap = eth_overlap
        self.eth_representation = eth_representation
        self.eth_image_size = eth_image_size
        
        self.eth_step = max(1, eth_window_size - eth_overlap)
        self.can_step = max(1, can_window_size - can_overlap)
        self.label_policy = label_policy

        can_df = pd.read_csv(can_csv_path)
        row_start = max(int(can_row_start or 0), 0)
        row_stop = None if can_row_stop is None else int(can_row_stop)
        if row_start or row_stop is not None:
            can_df = can_df.iloc[row_start:row_stop].reset_index(drop=True)
        if can_max_rows is not None:
            can_df = can_df.head(can_max_rows)
        self.can_timestamp_source = can_csv_path

        if "Timestamp" not in can_df.columns:
            can_timestamps, self.can_timestamp_source = _load_can_timestamps(
                can_csv_path,
                len(can_df),
                row_start=row_start,
                row_stop=row_stop,
            )
            can_df = standardize_can_dataframe(can_df, timestamp_values=can_timestamps)
        else:
            can_df = standardize_can_dataframe(can_df)

        raw_can_features = {"CAN_ID", "DLC", "Label", "Timestamp", "schema_version", *{f"D{i}" for i in range(8)}}
        needs_engineering = any(feature not in raw_can_features for feature in can_features)
        if needs_engineering:
            can_df = add_can_engineered_features(can_df)
        missing_can_features = [feature for feature in can_features if feature not in can_df.columns]
        if missing_can_features:
            raise ValueError(f"Missing CAN features after engineering: {missing_can_features}")
            
        self.can_values = can_df[can_features].to_numpy(dtype=np.float32)
        self.can_labels = can_df['Label'].astype(int).to_numpy()

        if eth_representation == STANDARD_ETH_REPRESENTATION:
            self.eth_images = build_eth_image_windows(
                eth_packet_csv_path=eth_packet_csv_path,
                eth_window_size=eth_window_size,
                eth_overlap=eth_overlap,
                image_size=eth_image_size,
                max_windows=eth_max_frames,
            )
        else:
            if not eth_npy_path:
                raise ValueError("eth_npy_path is required for legacy ETH image loading")
            eth_npy = np.load(eth_npy_path, mmap_mode='r')
            if eth_max_frames is not None and eth_npy.shape[0] > eth_max_frames:
                self.eth_images = eth_npy[:eth_max_frames]
            else:
                self.eth_images = eth_npy

        self.eth_labels, self.eth_label_source = _load_eth_labels(
            eth_packet_csv_path=eth_packet_csv_path,
            eth_npy_path=eth_npy_path or eth_packet_csv_path,
            eth_label_csv_path=eth_label_csv_path,
        )

        pairs_df, alignment_report = correlate_can_eth(
            can_csv_path=can_csv_path,
            eth_csv_path=eth_packet_csv_path,
            can_window_size=can_window_size,
            can_overlap=can_overlap,
            eth_window_size=eth_window_size,
            eth_overlap=eth_overlap,
            tolerance_ms=tolerance_ms,
            time_mode=time_mode,
            can_df=can_df,
        )

        max_eth_windows_from_images = 0
        if self.eth_images.shape[0] >= self.eth_window_size:
            max_eth_windows_from_images = (self.eth_images.shape[0] - self.eth_window_size) // self.eth_step + 1
            
        max_can_windows_from_values = 0
        if self.can_values.shape[0] >= self.can_window_size:
            max_can_windows_from_values = (self.can_values.shape[0] - self.can_window_size) // self.can_step + 1

        if max_eth_windows_from_images > 0 and max_can_windows_from_values > 0:
            pairs_df = pairs_df[
                (pairs_df['eth_window_idx'] < max_eth_windows_from_images) &
                (pairs_df['can_window_idx'] < max_can_windows_from_values)
            ].reset_index(drop=True)
        else:
            pairs_df = pairs_df.iloc[:0].copy()

        self.aligned_pairs = pairs_df
        self.alignment_report = alignment_report

        print(f"Initialized CorrelatedHybridVehicleDataset")
        print(f"Aligned pairs: {len(self.aligned_pairs)}")
        if hasattr(self.alignment_report, 'matched_rate_can'):
             print(f"Matched rate (CAN): {self.alignment_report.matched_rate_can:.4f}")
        print(f"CAN timestamp source: {os.path.basename(self.can_timestamp_source)}")
        print(f"ETH label source: {os.path.basename(self.eth_label_source)}")
        print(f"ETH representation: {self.eth_representation}")
        # Assuming alignment_report object attributes based on usage in disassembly
        if hasattr(self.alignment_report, 'median_delta_ms'):
             print(f"Median delta ms: {self.alignment_report.median_delta_ms}")
        if hasattr(self.alignment_report, 'p95_delta_ms'):
             print(f"P95 delta ms: {self.alignment_report.p95_delta_ms}")

    def __len__(self):
        return len(self.aligned_pairs)

    def _slice_can_window(self, can_window_idx: int) -> np.ndarray:
        start = can_window_idx * self.can_step
        end = start + self.can_window_size
        return self.can_values[start:end]

    def _eth_window_label(self, eth_window_idx: int) -> int:
        if self.eth_labels.size == 0:
            return 0
        if self.eth_labels.size == 1:
            return int(self.eth_labels[0])

        start = eth_window_idx * self.eth_step
        end = min(start + self.eth_window_size, len(self.eth_labels))
        if start >= len(self.eth_labels) or end <= start:
            return int(self.eth_labels[-1])
        return int(np.max(self.eth_labels[start:end]))

    def _slice_eth_window(self, eth_window_idx: int) -> np.ndarray:
        start = eth_window_idx * self.eth_step
        end = start + self.eth_window_size
        window = np.asarray(self.eth_images[start:end], dtype=np.float32)
        
        if window.size > 0 and window.max() > 1.5:
            window /= 255.0
            
        # Add channel dim if it's missing (N, H, W) -> (1, N, H, W) or (C, N, H, W)?
        # Disassembly says: expand_dims axis=1. 
        # Wait, if images are (N, H, W). Window is (W_size, H, W).
        # axis=1 -> (W_size, 1, H, W).
        return np.expand_dims(window, axis=1)

    def __getitem__(self, idx):
        row = self.aligned_pairs.iloc[idx]
        can_widx = int(row['can_window_idx'])
        eth_widx = int(row['eth_window_idx'])
        
        can_seq = self._slice_can_window(can_widx)
        eth_seq = self._slice_eth_window(eth_widx)
        
        # Calculate label for CAN window
        can_last_row_idx = can_widx * self.can_step + self.can_window_size - 1
        can_last_row_idx = min(can_last_row_idx, len(self.can_labels) - 1)
        can_label = int(self.can_labels[can_last_row_idx])
        eth_label = self._eth_window_label(eth_widx)
        
        combined_label = 0
        if self.label_policy == 'eth_only':
            combined_label = eth_label
        elif self.label_policy == 'can_only':
            combined_label = int(can_label)
        elif self.label_policy == 'and':
            combined_label = int(can_label and eth_label)
        else: # 'max' or default
            combined_label = max(can_label, eth_label)
            
        return {
            'can': torch.tensor(can_seq, dtype=torch.float32),
            'eth': torch.tensor(eth_seq, dtype=torch.float32),
            'label': torch.tensor(combined_label, dtype=torch.long),
            'can_window_idx': torch.tensor(can_widx, dtype=torch.long),
            'eth_window_idx': torch.tensor(eth_widx, dtype=torch.long),
            'delta_ms': torch.tensor(float(row['delta_ms']), dtype=torch.float32)
        }
