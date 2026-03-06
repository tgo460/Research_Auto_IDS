import os
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from correlation_replica import correlate_can_eth

class CorrelatedHybridVehicleDataset(Dataset):
    """
    Replica dataset that aligns CAN and ETH windows via timestamp/session correlation (C1).
    """
    def __init__(self, 
                 can_csv_path: str,
                 eth_packet_csv_path: str,
                 eth_npy_path: str,
                 can_features: List[str],
                 can_window_size: int = 100,
                 can_overlap: int = 50,
                 eth_window_size: int = 50,
                 eth_overlap: int = 25,
                 tolerance_ms: float = 100.0,
                 time_mode: str = 'relative_session',
                 can_max_rows: Optional[int] = None,
                 eth_max_frames: Optional[int] = None,
                 label_policy: str = 'max'):
        
        if not os.path.exists(can_csv_path):
            raise FileNotFoundError(f"CAN CSV not found: {can_csv_path}")
        if not os.path.exists(eth_packet_csv_path):
            raise FileNotFoundError(f"ETH packet CSV not found: {eth_packet_csv_path}")
        if not os.path.exists(eth_npy_path):
            raise FileNotFoundError(f"ETH image NPY not found: {eth_npy_path}")

        self.can_features = can_features
        self.can_window_size = can_window_size
        self.can_overlap = can_overlap
        self.eth_window_size = eth_window_size
        self.eth_overlap = eth_overlap
        
        self.eth_step = max(1, eth_window_size - eth_overlap)
        self.can_step = max(1, can_window_size - can_overlap)
        self.label_policy = label_policy

        can_df = pd.read_csv(can_csv_path)
        if can_max_rows is not None:
            can_df = can_df.head(can_max_rows)
            
        self.can_values = can_df[can_features].to_numpy(dtype=np.float32)
        self.can_labels = can_df['Label'].astype(int).to_numpy()

        eth_npy = np.load(eth_npy_path, mmap_mode='r')
        if eth_max_frames is not None and eth_npy.shape[0] > eth_max_frames:
            self.eth_images = eth_npy[:eth_max_frames]
        else:
            self.eth_images = eth_npy

        if 'injected' in os.path.basename(eth_npy_path).lower():
            self.eth_label = 1
        elif 'attack' in os.path.basename(eth_npy_path).lower():
            self.eth_label = 1
        else:
            self.eth_label = 0

        pairs_df, alignment_report = correlate_can_eth(
            can_csv_path=can_csv_path,
            eth_csv_path=eth_packet_csv_path,
            can_window_size=can_window_size,
            can_overlap=can_overlap,
            eth_window_size=eth_window_size,
            eth_overlap=eth_overlap,
            tolerance_ms=tolerance_ms,
            time_mode=time_mode
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

    def _slice_eth_window(self, eth_window_idx: int) -> np.ndarray:
        start = eth_window_idx * self.eth_step
        end = start + self.eth_window_size
        window = np.asarray(self.eth_images[start:end], dtype=np.float32)
        
        if window.max() > 1.5:
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
        
        combined_label = 0
        if self.label_policy == 'eth_only':
            combined_label = int(self.eth_label)
        elif self.label_policy == 'can_only':
            combined_label = int(can_label)
        elif self.label_policy == 'and':
            combined_label = int(can_label and self.eth_label)
        else: # 'max' or default
            combined_label = max(can_label, self.eth_label)
            
        return {
            'can': torch.tensor(can_seq, dtype=torch.float32),
            'eth': torch.tensor(eth_seq, dtype=torch.float32),
            'label': torch.tensor(combined_label, dtype=torch.long),
            'can_window_idx': torch.tensor(can_widx, dtype=torch.long),
            'eth_window_idx': torch.tensor(eth_widx, dtype=torch.long),
            'delta_ms': torch.tensor(float(row['delta_ms']), dtype=torch.float32)
        }
