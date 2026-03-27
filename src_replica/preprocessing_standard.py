from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Sequence

import numpy as np
import pandas as pd

BYTE_COLS = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

STANDARD_CAN_FEATURES_16 = [
    "CAN_ID",
    "DLC",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "can_id_freq_global",
    "can_id_freq_win",
    "payload_entropy",
    "inter_arrival",
    "inter_arrival_roll_mean",
    "id_switch_rate_win",
]

STANDARD_ETH_IMAGE_SIZE = 64  # upgraded from 32 to accommodate DPI payload features
STANDARD_ETH_REPRESENTATION = "metadata_outer_v1"
STANDARD_CAN_ENGINEERING_WINDOW = 200
STANDARD_ETH_ENGINEERING_WINDOW = 32
# Phase 1 DPI: payload byte columns extracted from PCAP
STANDARD_ETH_PAYLOAD_BYTES = 16
STANDARD_ETH_PAYLOAD_COLS = [f"payload_b{i}" for i in range(STANDARD_ETH_PAYLOAD_BYTES)]
STANDARD_ETH_DPI_COLS = ["eth_type", "ip_proto", "src_port", "dst_port", "payload_len", "payload_entropy"]
ETH_LABEL_PROVENANCE_COLUMNS = [
    "session_id",
    "attack_type",
    "label_source",
    "label_granularity",
]
TRUSTED_ETH_LABEL_SOURCES = {
    "packet_ground_truth",
    "attack_log_interval",
    "manual_packet_annotation",
    "manual_window_annotation",
}

_CAN_CLASSIC_ID_MAX = 2047.0
_CAN_EXTENDED_ID_MAX = 536_870_911.0
_CAN_CLASSIC_DLC_MAX = 8.0
_CAN_FD_DLC_MAX = 64.0
_ETH_PACKET_LEN_SCALE = 1518.0
_ETH_RATIO_MAX = 1.5
_SYNTHETIC_TIMESTAMP_STEP_S = 0.001


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _series_from_frame_or_scalar(df: pd.DataFrame, key: str, default: float = 0.0) -> pd.Series:
    if key in df.columns:
        value = df[key]
    else:
        value = pd.Series([default] * len(df), index=df.index, dtype=np.float32)
    return pd.to_numeric(value, errors="coerce").fillna(default)


def validate_eth_label_dataframe(
    df: pd.DataFrame,
    context: str = "ETH dataframe",
    require_label: bool = True,
    require_provenance: bool = False,
) -> None:
    missing = []
    if require_label and "Label" not in df.columns:
        missing.append("Label")
    if require_provenance:
        missing.extend(col for col in ETH_LABEL_PROVENANCE_COLUMNS if col not in df.columns)
    if missing:
        raise ValueError(
            f"{context} must contain required columns: {', '.join(missing)}"
        )


def extract_eth_label_provenance(df: pd.DataFrame) -> Dict[str, str]:
    provenance: Dict[str, str] = {}
    for col in ETH_LABEL_PROVENANCE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        provenance[col] = series.iloc[0] if not series.empty else ""
    return provenance


def is_trustworthy_eth_label_provenance(provenance: Dict[str, str] | None) -> bool:
    if not provenance:
        return False
    source = str(provenance.get("label_source", "")).strip()
    granularity = str(provenance.get("label_granularity", "")).strip()
    if source not in TRUSTED_ETH_LABEL_SOURCES:
        return False
    return granularity in {"packet", "window"}


def _running_max_normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    denom = np.maximum(np.maximum.accumulate(arr), 1e-9)
    return (arr / denom).astype(np.float32)


def normalize_can_id_series(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    arr = np.abs(arr)
    out = np.zeros_like(arr, dtype=np.float64)

    small_mask = arr <= 1.0
    classic_mask = (~small_mask) & (arr <= _CAN_CLASSIC_ID_MAX)
    extended_mask = (~small_mask) & (~classic_mask)

    out[small_mask] = arr[small_mask]
    out[classic_mask] = arr[classic_mask] / _CAN_CLASSIC_ID_MAX
    out[extended_mask] = arr[extended_mask] / _CAN_EXTENDED_ID_MAX
    return pd.Series(_clip01(out.astype(np.float32)), index=series.index)


def normalize_dlc_series(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    positive = arr[arr > 0.0]
    if positive.size > 0 and float(positive.max()) <= ((_CAN_CLASSIC_DLC_MAX / 255.0) + 1e-3):
        out = np.clip((arr * 255.0) / _CAN_CLASSIC_DLC_MAX, 0.0, 1.0)
        return pd.Series(out.astype(np.float32), index=series.index)

    out = np.zeros_like(arr, dtype=np.float64)

    normalized_mask = arr <= 1.0
    classic_mask = (~normalized_mask) & (arr <= _CAN_CLASSIC_DLC_MAX)
    fd_mask = (~normalized_mask) & (~classic_mask)

    out[normalized_mask] = arr[normalized_mask]
    out[classic_mask] = arr[classic_mask] / _CAN_CLASSIC_DLC_MAX
    out[fd_mask] = np.minimum(arr[fd_mask], _CAN_FD_DLC_MAX) / _CAN_FD_DLC_MAX
    return pd.Series(_clip01(out.astype(np.float32)), index=series.index)


def normalize_payload_series(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    normalized_mask = (arr >= 0.0) & (arr <= 1.0)
    out = np.where(normalized_mask, arr, arr / 255.0)
    return pd.Series(_clip01(out.astype(np.float32)), index=series.index)


def standardize_can_dataframe(
    df_raw: pd.DataFrame,
    timestamp_values: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    out = df_raw.copy()

    if "CAN_ID" not in out.columns or "DLC" not in out.columns:
        raise ValueError("CAN dataframe must contain CAN_ID and DLC columns")
    for col in BYTE_COLS:
        if col not in out.columns:
            out[col] = 0.0

    out["CAN_ID"] = normalize_can_id_series(out["CAN_ID"])
    out["DLC"] = normalize_dlc_series(out["DLC"])
    for col in BYTE_COLS:
        out[col] = normalize_payload_series(out[col])

    if timestamp_values is not None and "Timestamp" not in out.columns:
        out["Timestamp"] = np.asarray(timestamp_values, dtype=np.float64)
    if "Timestamp" in out.columns:
        out["Timestamp"] = pd.to_numeric(out["Timestamp"], errors="coerce").ffill().fillna(0.0)
    else:
        out["Timestamp"] = np.arange(len(out), dtype=np.float64) * _SYNTHETIC_TIMESTAMP_STEP_S

    if "Label" in out.columns:
        out["Label"] = pd.to_numeric(out["Label"], errors="coerce").fillna(0).astype(int)
    else:
        out["Label"] = 0

    return out


def _payload_entropy_normalized(payload_row: np.ndarray) -> float:
    payload_i = np.clip(np.asarray(payload_row, dtype=np.float32) * 255.0, 0, 255).astype(np.int32)
    hist, _ = np.histogram(payload_i, bins=16, range=(0, 256), density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(min(len(payload_i), 16)) if len(payload_i) > 1 else 1.0
    return 0.0 if max_entropy <= 0 else float(entropy / max_entropy)


def _rolling_id_frequency(can_ids: np.ndarray, window: int) -> np.ndarray:
    counts: Dict[float, int] = {}
    hist: Deque[float] = deque()
    out = np.zeros(len(can_ids), dtype=np.float32)
    for idx, can_id in enumerate(can_ids.tolist()):
        hist.append(can_id)
        counts[can_id] = counts.get(can_id, 0) + 1
        if len(hist) > window:
            old = hist.popleft()
            counts[old] -= 1
            if counts[old] <= 0:
                counts.pop(old, None)
        out[idx] = counts[can_id] / float(len(hist))
    return out


def add_standard_can_features(
    df_raw: pd.DataFrame,
    window: int = STANDARD_CAN_ENGINEERING_WINDOW,
) -> pd.DataFrame:
    out = standardize_can_dataframe(df_raw)
    can_ids = out["CAN_ID"].round(6)

    prefix_count = can_ids.groupby(can_ids).cumcount().to_numpy(dtype=np.float32) + 1.0
    positions = np.arange(1, len(out) + 1, dtype=np.float32)
    out["can_id_freq_global"] = prefix_count / positions

    can_id_values = can_ids.to_numpy(dtype=np.float64)
    out["can_id_freq_win"] = _rolling_id_frequency(can_id_values, window=window)

    payload = out[BYTE_COLS].to_numpy(dtype=np.float32)
    out["payload_entropy"] = np.asarray(
        [_payload_entropy_normalized(row) for row in payload],
        dtype=np.float32,
    )

    ts = pd.to_numeric(out["Timestamp"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float64)
    inter_arrival_raw = np.diff(ts, prepend=ts[:1]).astype(np.float32)
    inter_arrival_raw = np.clip(inter_arrival_raw, 0.0, None)
    if inter_arrival_raw.size:
        inter_arrival_raw[0] = 0.0
    out["inter_arrival"] = _running_max_normalize(inter_arrival_raw)

    inter_arrival_roll = (
        pd.Series(out["inter_arrival"], dtype=np.float32)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    out["inter_arrival_roll_mean"] = _clip01(inter_arrival_roll)

    switches = (~can_ids.diff().fillna(0.0).abs().lt(1e-6)).astype(np.float32)
    if len(switches) > 0:
        switches.iloc[0] = 0.0
    switch_rate = (
        switches.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float32)
    )
    out["id_switch_rate_win"] = _clip01(switch_rate)

    return out


def standardize_eth_packet_dataframe(
    df_raw: pd.DataFrame,
    rolling_window: int = STANDARD_ETH_ENGINEERING_WINDOW,
) -> pd.DataFrame:
    out = df_raw.copy()

    if "captured_len" in out.columns:
        captured = _series_from_frame_or_scalar(out, "captured_len", default=0.0)
        original = _series_from_frame_or_scalar(out, "original_len", default=0.0)
        if "original_len" not in out.columns:
            original = captured.copy()
    else:
        packet_len = _series_from_frame_or_scalar(out, "Packet_Length", default=0.0)
        captured = packet_len
        original = packet_len

    if "timestamp_sec" in out.columns:
        sec = _series_from_frame_or_scalar(out, "timestamp_sec", default=0.0).to_numpy(dtype=np.float64)
        usec = _series_from_frame_or_scalar(out, "timestamp_usec", default=0.0).to_numpy(dtype=np.float64)
        finite = sec[np.isfinite(sec)]
        if finite.size > 0:
            med = float(np.median(np.abs(finite)))
            if med > 1e17:
                sec = sec / 1e9
            elif med > 1e14:
                sec = sec / 1e6
            elif med > 1e11:
                sec = sec / 1e3
        timestamp = sec + (usec / 1_000_000.0)
    elif "timestamp" in out.columns:
        timestamp = pd.to_numeric(out["timestamp"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float64)
    else:
        timestamp = np.arange(len(out), dtype=np.float64) * _SYNTHETIC_TIMESTAMP_STEP_S

    timestamp = np.asarray(timestamp, dtype=np.float64)
    inter_arrival_raw = np.diff(timestamp, prepend=timestamp[:1]).astype(np.float32)
    inter_arrival_raw = np.clip(inter_arrival_raw, 0.0, None)
    if inter_arrival_raw.size:
        inter_arrival_raw[0] = 0.0
    inter_arrival_norm = _running_max_normalize(inter_arrival_raw)

    captured_arr = captured.to_numpy(dtype=np.float32)
    original_arr = original.replace(0.0, np.nan).to_numpy(dtype=np.float32)
    length_ratio = np.divide(
        captured_arr,
        original_arr,
        out=np.zeros_like(captured_arr, dtype=np.float32),
        where=np.isfinite(original_arr) & (original_arr > 0.0),
    )
    length_ratio = np.clip(length_ratio, 0.0, _ETH_RATIO_MAX) / _ETH_RATIO_MAX

    captured_norm = np.clip(captured_arr / _ETH_PACKET_LEN_SCALE, 0.0, 1.0)
    original_norm = np.clip(
        np.nan_to_num(original_arr, nan=0.0, posinf=0.0, neginf=0.0) / _ETH_PACKET_LEN_SCALE,
        0.0,
        1.0,
    )
    packet_delta_norm = np.zeros_like(captured_norm, dtype=np.float32)
    if len(captured_norm) > 1:
        packet_delta_norm[1:] = np.clip(
            np.abs(np.diff(captured_norm)),
            0.0,
            1.0,
        )

    roll_len_mean = (
        pd.Series(captured_norm)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    roll_gap_mean = (
        pd.Series(inter_arrival_norm)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )

    standardized = pd.DataFrame(
        {
            "timestamp": timestamp,
            "captured_len": captured_arr,
            "original_len": np.nan_to_num(original_arr, nan=0.0, posinf=0.0, neginf=0.0),
            "captured_len_norm": captured_norm,
            "original_len_norm": original_norm,
            "length_ratio_norm": length_ratio.astype(np.float32),
            "inter_arrival_norm": inter_arrival_norm.astype(np.float32),
            "rolling_len_mean_norm": _clip01(roll_len_mean),
            "rolling_gap_mean_norm": _clip01(roll_gap_mean),
            "packet_delta_norm": packet_delta_norm.astype(np.float32),
        }
    )

    if "Label" in out.columns:
        standardized["Label"] = pd.to_numeric(out["Label"], errors="coerce").fillna(0).astype(int)
    for col in ETH_LABEL_PROVENANCE_COLUMNS:
        if col in out.columns:
            standardized[col] = out[col].fillna("").astype(str)

    # ── Phase 1 DPI: normalise and carry payload columns forward ──────────────
    # Payload byte features (payload_b0 … payload_b15) — raw byte values [0,255] -> [0,1]
    for col in STANDARD_ETH_PAYLOAD_COLS:
        if col in out.columns:
            raw_vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            # Values from DPI are already pre-normalised to [0,1]; if raw bytes remain, divide by 255
            if raw_vals.max() > 1.5:
                raw_vals = raw_vals / 255.0
            standardized[col] = _clip01(raw_vals)
        else:
            standardized[col] = np.zeros(len(standardized), dtype=np.float32)

    # Scalar DPI features
    for col in ["eth_type", "ip_proto", "src_port", "dst_port"]:
        if col in out.columns:
            standardized[col] = _clip01(
                pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            )
        else:
            standardized[col] = np.zeros(len(standardized), dtype=np.float32)

    if "payload_entropy" in out.columns:
        standardized["payload_entropy"] = _clip01(
            pd.to_numeric(out["payload_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        )
    else:
        standardized["payload_entropy"] = np.zeros(len(standardized), dtype=np.float32)

    if "payload_len" in out.columns:
        # Normalise by max Ethernet payload (1500 bytes)
        standardized["payload_len_norm"] = _clip01(
            pd.to_numeric(out["payload_len"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) / 1500.0
        )
    else:
        standardized["payload_len_norm"] = np.zeros(len(standardized), dtype=np.float32)

    return standardized


def _eth_window_summary(window_df: pd.DataFrame) -> np.ndarray:
    cap = window_df["captured_len_norm"].to_numpy(dtype=np.float32)
    orig = window_df["original_len_norm"].to_numpy(dtype=np.float32)
    ratio = window_df["length_ratio_norm"].to_numpy(dtype=np.float32)
    gap = window_df["inter_arrival_norm"].to_numpy(dtype=np.float32)
    roll_len = window_df["rolling_len_mean_norm"].to_numpy(dtype=np.float32)
    roll_gap = window_df["rolling_gap_mean_norm"].to_numpy(dtype=np.float32)
    delta = window_df["packet_delta_norm"].to_numpy(dtype=np.float32)

    # ── Metadata features (10 values — same as before) ───────────────────────
    meta_features = np.asarray(
        [
            float(cap.mean()),
            float(cap.std(ddof=0)),
            float(cap.min()),
            float(cap.max()),
            float(orig.mean()),
            float(orig.std(ddof=0)),
            float(ratio.mean()),
            float(ratio.std(ddof=0)),
            float(gap.mean()),
            float(gap.std(ddof=0)),
            float(gap.max()),
            float(roll_len.mean()),
            float(roll_gap.mean()),
            float(delta.mean()),
            float(cap[-1]),
            float(ratio[-1]),
        ],
        dtype=np.float32,
    )

    # ── Phase 1 DPI: payload-derived features (20 values) ────────────────────
    # 16 payload byte means over the window
    payload_byte_means = np.zeros(STANDARD_ETH_PAYLOAD_BYTES, dtype=np.float32)
    for i, col in enumerate(STANDARD_ETH_PAYLOAD_COLS):
        if col in window_df.columns:
            payload_byte_means[i] = float(window_df[col].to_numpy(dtype=np.float32).mean())

    # Payload entropy aggregates (mean, std, max)
    if "payload_entropy" in window_df.columns:
        ent = window_df["payload_entropy"].to_numpy(dtype=np.float32)
        payload_entropy_mean = float(ent.mean())
        payload_entropy_std = float(ent.std(ddof=0))
        payload_entropy_max = float(ent.max())
    else:
        payload_entropy_mean = payload_entropy_std = payload_entropy_max = 0.0

    # Network header features (ip_proto, src_port mean, dst_port mean, payload_len)
    ip_proto_mean = float(window_df["ip_proto"].mean()) if "ip_proto" in window_df.columns else 0.0
    dst_port_mean = float(window_df["dst_port"].mean()) if "dst_port" in window_df.columns else 0.0

    dpi_features = np.asarray(
        [
            *payload_byte_means.tolist(),      # 16 values
            payload_entropy_mean,              # 1
            payload_entropy_std,               # 1
            payload_entropy_max,               # 1
            ip_proto_mean,                     # 1
        ],
        dtype=np.float32,
    )  # total: 20 DPI values

    features = np.concatenate([meta_features, dpi_features]).astype(np.float32)  # 16 + 20 = 36 values
    return _clip01(features)


def encode_eth_window_to_image(
    window_df: pd.DataFrame,
    image_size: int = STANDARD_ETH_IMAGE_SIZE,
) -> np.ndarray:
    stats = _eth_window_summary(window_df)  # 36-element vector
    # Pad or truncate to half the image_size so that outer product fills image_size x image_size
    half = image_size // 2
    if len(stats) < half:
        stats = np.pad(stats, (0, half - len(stats)), constant_values=0.0)
    else:
        stats = stats[:half]
    vector = np.concatenate([stats, 1.0 - stats], axis=0).astype(np.float32)  # image_size elements
    image = np.outer(vector, vector).astype(np.float32)
    return _clip01(image)


def build_eth_image_windows(
    eth_packet_csv_path: str,
    eth_window_size: int,
    eth_overlap: int = 0,
    image_size: int = STANDARD_ETH_IMAGE_SIZE,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    raw_df = pd.read_csv(eth_packet_csv_path)
    standardized = standardize_eth_packet_dataframe(raw_df)

    step = max(1, eth_window_size - eth_overlap)
    if len(standardized) < eth_window_size:
        return np.zeros((0, image_size, image_size), dtype=np.float32)

    windows = []
    for start in range(0, len(standardized) - eth_window_size + 1, step):
        end = start + eth_window_size
        windows.append(encode_eth_window_to_image(standardized.iloc[start:end], image_size=image_size))
        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        return np.zeros((0, image_size, image_size), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def encode_eth_frame_dict_to_image(
    frame: Dict[str, float],
    image_size: int = STANDARD_ETH_IMAGE_SIZE,
) -> np.ndarray:
    df = pd.DataFrame(
        [
            {
                "timestamp": float(frame.get("timestamp", 0.0)),
                "captured_len": float(frame.get("captured_len", 0.0)),
                "original_len": float(frame.get("original_len", frame.get("captured_len", 0.0))),
            }
        ]
    )
    standardized = standardize_eth_packet_dataframe(df)
    return encode_eth_window_to_image(standardized, image_size=image_size)
