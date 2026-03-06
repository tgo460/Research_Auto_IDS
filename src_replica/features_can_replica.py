import math
from typing import List
import numpy as np
import pandas as pd

BYTE_COLS: List[str] = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

def _safe_timestamp_delta(ts: pd.Series) -> pd.Series:
    ts_num = pd.to_numeric(ts, errors='coerce').ffill().fillna(0.0)
    return ts_num.diff().fillna(0.0).clip(lower=0.0)

def _row_entropy(byte_values: np.ndarray) -> float:
    hist, _ = np.histogram(byte_values, bins=16, range=(0, 256), density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist / total
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def add_can_engineered_features(df_raw: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    out = df_raw.copy()
    can_id_counts = out['CAN_ID'].astype(int).value_counts().to_dict()
    out['can_id_freq_global'] = out['CAN_ID'].astype(int).map(can_id_counts).astype(float)
    max_freq = max(float(out['can_id_freq_global'].max()), 1.0)
    out['can_id_freq_global'] = out['can_id_freq_global'] / max_freq

    out['can_id_freq_win'] = out['CAN_ID'].astype(int).rolling(window=window, min_periods=1).apply(
        lambda x: (x.values == x.values[-1]).sum() / max(len(x), 1),
        raw=False
    )

    payload = out[BYTE_COLS].to_numpy(dtype=np.float32)
    
    entropy_list = []
    for row in payload:
        # Reconstruct byte values from float scale if needed or just use as is
        # The logic in disasm shows clipping * 255.0 astype int32 clip 0 255
        # row is float32
        row_int = np.clip(row * 255.0, 0, 255).astype(np.int32)
        entropy_list.append(_row_entropy(row_int))
    
    out['payload_entropy'] = entropy_list

    max_ent = max(float(out['payload_entropy'].max()), 1.0)
    out['payload_entropy'] = out['payload_entropy'] / max_ent

    inter_arrival = _safe_timestamp_delta(out['Timestamp'])
    ia_max = max(float(inter_arrival.max()), 1.0)
    out['inter_arrival'] = inter_arrival / ia_max

    out['inter_arrival_roll_mean'] = out['inter_arrival'].rolling(window=window, min_periods=1).mean()

    switches = (out['CAN_ID'].astype(int).diff().fillna(0) != 0).astype(float)
    out['id_switch_rate_win'] = switches.rolling(window=window, min_periods=1).mean()

    return out
