import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

CSV_PATTERN = re.compile(r'^eth_(.+?)_replica_packets\.csv$')
NPY_PATTERN = re.compile(r'^eth_(.+?)_images(?:[-_].+)?\.npy$')

@dataclass
class ScenarioRows:
    key: str
    label: int
    protocol_df: pd.DataFrame
    image_df: pd.DataFrame

def scenario_key_from_protocol_csv(file_name: str) -> Optional[str]:
    m = CSV_PATTERN.match(file_name)
    if m:
        return m.group(1)
    return None

def scenario_key_from_image_npy(file_name: str) -> Optional[str]:
    m = NPY_PATTERN.match(file_name)
    if m:
        return m.group(1)
    return None

def label_from_key(key: str) -> int:
    key_l = key.lower()
    if 'injected' in key_l:
        return 1
    # Check disassembly again for exact logic.
    # 34 'injected' in key_l -> JUMP L1 (return 1)
    # 'attack' in key_l -> JUMP L2 (return 0), if false fall through presumably?
    # Wait, the disassembly snippet is:
    # 34 'injected' in key_l -> if true jump to L1 which returns 1.
    # 'attack' in key_l -> if false jump to L2 which returns 0.
    # if true (attack in key), it falls through to L1? NO.
    # Let's re-read carefully.
    # ... POP_JUMP_IF_TRUE 6 (to L1)
    # ... POP_JUMP_IF_FALSE 2 (to L2)
    # L1: RETURN 1
    # L2: RETURN 0
    # So if injected is in key -> 1.
    # If injected NOT in key, check attack.
    # If attack IS in key -> fallthrough to L1 -> return 1.
    # If attack NOT in key -> jump to L2 -> return 0.
    if 'attack' in key_l:
        return 1
    return 0

def build_protocol_window_features(packet_df: pd.DataFrame, window_size: int, max_windows: Optional[int]) -> pd.DataFrame:
    # 42
    ts = packet_df['timestamp_sec'].astype(float) + packet_df['timestamp_usec'].astype(float) / 1000000.0
    # 43
    gaps = ts.diff().fillna(0.0).clip(lower=0.0)
    # 44
    cap_len = packet_df['captured_len'].astype(float)
    # 45
    orig_len = packet_df['original_len'].astype(float).replace(0.0, np.nan)
    # 46
    eff = (cap_len / orig_len).fillna(0.0).clip(lower=0.0, upper=1.5)
    
    rows = []
    total = len(packet_df)
    n_windows = total // window_size
    
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
        
    for w in range(n_windows):
        s = w * window_size
        e = s + window_size
        
        p_len = cap_len.iloc[s:e]
        p_gap = gaps.iloc[s:e]
        # p_eff = eff.iloc[s:e] # Disassembled code computes it but maybe doesn't use it or I missed it?
        # Looking at L67 appending to rows:
        # w, mean_len, std_len, min_len, max_len, (mean_len>0 ? std/mean : 0), mean_gap, std_gap
        # Wait, let's trace the append carefully.
        
        mean_len = float(p_len.mean())
        std_len = float(p_len.std(ddof=0))
        mean_gap = float(p_gap.mean())
        std_gap = float(p_gap.std(ddof=0))
        
        # Disassembly lines 67-74
        # append(w)
        # append(mean_len)
        # append(std_len)
        # append(p_len.min())
        # append(p_len.max())
        # if mean_len > 0: std_len / mean_len else 0.0
        # append(mean_gap)
        # append(std_gap)
        # It seems to be building a list/tuple to append.
        
        # Re-check the column order from context or inference.
        # Usually: window_idx, len_mean, len_std, len_min, len_max, len_cov, gap_mean, gap_std...
        
        cov = std_len / mean_len if mean_len > 0 else 0.0
        
        rows.append({
            'window_idx': w,
            'len_mean': mean_len,
            'len_std': std_len,
            'len_min': float(p_len.min()),
            'len_max': float(p_len.max()),
            'len_cov': cov,
            'gap_mean': mean_gap,
            'gap_std': std_gap
        })
        
    return pd.DataFrame(rows)



def build_image_window_features(npy_path: str, window_size: int, max_windows: Optional[int]) -> pd.DataFrame:
    try:
        arr = np.load(npy_path, mmap_mode='r')
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return pd.DataFrame()
        
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image tensor (N,H,W), got shape={arr.shape} for {npy_path}")

    total = int(arr.shape[0])
    n_windows = total // window_size
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    rows = []
    for w in range(n_windows):
        s = w * window_size
        e = s + window_size
        
        # Ensure we don't go out of bounds (though range calculation should prevent it)
        if e > total:
            break
            
        win = np.asarray(arr[s:e]).astype(np.float32)
        
        if win.max() > 1.5:
            win /= 255.0
            
        mean_val = float(win.mean())
        std_val = float(win.std())
        nonzero = float((win > 0).mean())
        p95 = float(np.percentile(win, 95))
        
        # Gradients
        if win.shape[2] > 1:
            gx = np.abs(np.diff(win, axis=2)).mean()
        else:
            gx = 0.0
            
        if win.shape[1] > 1:
            gy = np.abs(np.diff(win, axis=1)).mean()
        else:
            gy = 0.0
            
        if win.shape[0] > 1:
            gt = np.abs(np.diff(win, axis=0)).mean()
        else:
            gt = 0.0
            
        rows.append({
            'window_idx': w,
            'i_mean': mean_val,
            'i_std': std_val,
            'i_nonzero_ratio': nonzero,
            'i_p95': p95,
            'i_grad_spatial': float((gx + gy) / 2.0),
            'i_grad_temporal': float(gt)
        })
        
    return pd.DataFrame(rows)

def discover_eth_sources(protocol_dir: str, image_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    protocol_map = {}
    if os.path.exists(protocol_dir):
        for name in sorted(os.listdir(protocol_dir)):
            key = scenario_key_from_protocol_csv(name)
            if key:
                protocol_map[key] = os.path.join(protocol_dir, name)
                
    image_map = {}
    if os.path.exists(image_dir):
        for name in sorted(os.listdir(image_dir)):
            key = scenario_key_from_image_npy(name)
            if key:
                image_map[key] = os.path.join(image_dir, name)
                
    return protocol_map, image_map

def build_eth_feature_tables(protocol_dir: str, image_dir: str, window_size: int = 20, max_windows_per_scenario: Optional[int] = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol_map, image_map = discover_eth_sources(protocol_dir, image_dir)
    
    shared_keys = set(protocol_map.keys()) & set(image_map.keys())
    if not shared_keys:
        # It's possible we want to proceed even if empty, but the disassembly raises error
        raise FileNotFoundError('No shared ETH scenarios between protocol CSVs and image NPY files.')
        
    protocol_rows = []
    image_rows = []
    combined_rows = []
    
    for key in shared_keys:
        label = label_from_key(key)
        
        p_df_raw = pd.read_csv(protocol_map[key])
        p_feat = build_protocol_window_features(p_df_raw, window_size, max_windows_per_scenario)
        
        i_feat = build_image_window_features(image_map[key], window_size, max_windows_per_scenario)
        
        max_common = min(len(p_feat), len(i_feat))
        if max_common <= 0:
            continue
            
        p_feat = p_feat.iloc[:max_common].copy()
        i_feat = i_feat.iloc[:max_common].copy()
        
        p_feat['scenario_key'] = key
        p_feat['label'] = label
        i_feat['scenario_key'] = key
        i_feat['label'] = label
        
        merged = pd.merge(
            p_feat, 
            i_feat, 
            on=['scenario_key', 'window_idx', 'label'], 
            how='inner', 
            suffixes=('_p', '_i')
        )
        
        if merged.empty:
            continue
            
        protocol_rows.append(p_feat)
        image_rows.append(i_feat)
        combined_rows.append(merged)
        
    if not combined_rows:
        raise ValueError('No aligned ETH windows produced for ablation.')
        
    protocol_all = pd.concat(protocol_rows, ignore_index=True)
    image_all = pd.concat(image_rows, ignore_index=True)
    combined_all = pd.concat(combined_rows, ignore_index=True)
    
    return protocol_all, image_all, combined_all

