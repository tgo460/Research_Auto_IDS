import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class AlignmentResult:
    pair_name: str
    can_windows: int
    eth_windows: int
    matched: int
    unmatched_can: int
    unmatched_eth_est: int
    matched_rate_can: float
    unmatched_rate_can: float
    median_delta_ms: Optional[float]
    p95_delta_ms: Optional[float]
    tolerance_ms: float
    time_mode: str

def _infer_epoch_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    med = float(np.median(np.abs(finite)))
    if med > 1e17:
        return 1e9
    if med > 1e14:
        return 1e6
    if med > 1e11:
        return 1e3
    return 1.0

def normalize_can_timestamps(can_df: pd.DataFrame) -> np.ndarray:
    if 'Timestamp' in can_df.columns:
        ts = pd.to_numeric(can_df['Timestamp'], errors='coerce').ffill().fillna(0.0).to_numpy(dtype=np.float64)
    else:
        ts = np.arange(len(can_df), dtype=np.float64) * 0.001
    
    scale = _infer_epoch_scale(ts)
    return ts / scale

def normalize_eth_timestamps(eth_df: pd.DataFrame) -> np.ndarray:
    sec_like = pd.to_numeric(eth_df['timestamp_sec'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    usec_like = pd.to_numeric(eth_df.get('timestamp_usec', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    
    scale = _infer_epoch_scale(sec_like)
    sec_base = sec_like / scale
    return sec_base + (usec_like / 1000000.0)

def make_window_end_timestamps(ts_seconds: np.ndarray, window_size: int, overlap: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    
    step = max(1, window_size - overlap)
    if ts_seconds.size < window_size:
        return np.array([], dtype=np.float64)
    
    end_idx = np.arange(window_size - 1, ts_seconds.size, step)
    return ts_seconds[end_idx]

def apply_time_mode(can_ts: np.ndarray, eth_ts: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == 'absolute':
        return can_ts, eth_ts
    elif mode == 'relative_session':
        can0 = can_ts[0] if can_ts.size else 0.0
        eth0 = eth_ts[0] if eth_ts.size else 0.0
        return can_ts - can0, eth_ts - eth0
    else:
        raise ValueError("time_mode must be one of: absolute, relative_session")

def nearest_align(
    can_windows_ts: np.ndarray,
    eth_windows_ts: np.ndarray,
    tolerance_ms: float
) -> pd.DataFrame:
    if can_windows_ts.size == 0 or eth_windows_ts.size == 0:
        return pd.DataFrame(columns=['can_window_idx', 'eth_window_idx', 'can_ts', 'eth_ts', 'delta_ms'])
        
    rows = []
    tol_s = tolerance_ms / 1000.0
    
    eth_sorted_idx = np.argsort(eth_windows_ts)
    eth_sorted_ts = eth_windows_ts[eth_sorted_idx]
    
    for i, t_can in enumerate(can_windows_ts):
        pos = np.searchsorted(eth_sorted_ts, t_can)
        candidates = []
        
        if pos < eth_sorted_ts.size:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
            
        if candidates:
            best_pos = min(candidates, key=lambda p: abs(eth_sorted_ts[p] - t_can))
            delta_s = abs(eth_sorted_ts[best_pos] - t_can)
            
            if delta_s <= tol_s:
                eth_idx = int(eth_sorted_idx[best_pos])
                rows.append({
                    'can_window_idx': int(i),
                    'eth_window_idx': eth_idx,
                    'can_ts': float(t_can),
                    'eth_ts': float(eth_windows_ts[eth_idx]),
                    'delta_ms': float(delta_s * 1000.0)
                })
                
    return pd.DataFrame(rows)

def build_alignment_report(
    pair_name: str,
    can_windows_ts: np.ndarray,
    eth_windows_ts: np.ndarray,
    pairs_df: pd.DataFrame,
    tolerance_ms: float,
    time_mode: str
) -> AlignmentResult:
    matched = int(len(pairs_df))
    can_total = int(len(can_windows_ts))
    eth_total = int(len(eth_windows_ts))
    
    unmatched_can = max(0, can_total - matched)
    
    if matched > 0:
        matched_eth_unique = int(pairs_df['eth_window_idx'].nunique())
    else:
        matched_eth_unique = 0
        
    unmatched_eth_est = max(0, eth_total - matched_eth_unique)
    
    if matched > 0:
        median_delta = float(pairs_df['delta_ms'].median())
    else:
        median_delta = None
        
    if matched > 0:
        p95_delta = float(np.percentile(pairs_df['delta_ms'], 95))
    else:
        p95_delta = None
        
    if can_total > 0:
        matched_rate_can = float(matched / can_total)
    else:
        matched_rate_can = 0.0
        
    if can_total > 0:
        unmatched_rate_can = float(unmatched_can / can_total)
    else:
        unmatched_rate_can = 0.0
        
    return AlignmentResult(
        pair_name=pair_name,
        can_windows=can_total,
        eth_windows=eth_total,
        matched=matched,
        unmatched_can=unmatched_can,
        unmatched_eth_est=unmatched_eth_est,
        matched_rate_can=matched_rate_can,
        unmatched_rate_can=unmatched_rate_can,
        median_delta_ms=median_delta,
        p95_delta_ms=p95_delta,
        tolerance_ms=float(tolerance_ms),
        time_mode=time_mode
    )

def correlate_can_eth(
    can_csv_path: str,
    eth_csv_path: str,
    can_window_size: int,
    can_overlap: int,
    eth_window_size: int,
    eth_overlap: int,
    tolerance_ms: float,
    time_mode: str = 'relative_session'
) -> Tuple[pd.DataFrame, AlignmentResult]:
    
    can_df = pd.read_csv(can_csv_path)
    eth_df = pd.read_csv(eth_csv_path)
    
    can_ts = normalize_can_timestamps(can_df)
    eth_ts = normalize_eth_timestamps(eth_df)
    
    can_ts, eth_ts = apply_time_mode(can_ts, eth_ts, mode=time_mode)
    
    can_win_ts = make_window_end_timestamps(can_ts, can_window_size, can_overlap)
    eth_win_ts = make_window_end_timestamps(eth_ts, eth_window_size, eth_overlap)
    
    pairs_df = nearest_align(can_win_ts, eth_win_ts, tolerance_ms=tolerance_ms)
    
    pair_name = f"{os.path.basename(can_csv_path)}__{os.path.basename(eth_csv_path)}"
    
    report = build_alignment_report(
        pair_name=pair_name,
        can_windows_ts=can_win_ts,
        eth_windows_ts=eth_win_ts,
        pairs_df=pairs_df,
        tolerance_ms=tolerance_ms,
        time_mode=time_mode
    )
    
    return pairs_df, report

def report_to_dict(result: AlignmentResult) -> Dict[str, object]:
    return asdict(result)

def report_to_dict(result: AlignmentResult) -> Dict[str, object]:
    return asdict(result)
