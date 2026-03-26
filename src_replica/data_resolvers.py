import csv
import os
from typing import Optional


def _eth_base_name(eth_npy_file: str) -> str:
    if "_images" in eth_npy_file:
        return eth_npy_file.split("_images")[0]
    return os.path.splitext(eth_npy_file)[0]


def resolve_can_csv(data_dir: str, can_file: str, prefer_raw: bool = True) -> Optional[str]:
    candidates = []
    raw_path = os.path.join(data_dir, can_file)
    engineered_path = os.path.join(data_dir, "replica_can_b1_engineered", can_file)
    if prefer_raw:
        candidates.extend([raw_path, engineered_path])
    else:
        candidates.extend([engineered_path, raw_path])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_eth_packet_csv(data_dir: str, eth_npy_file: str) -> Optional[str]:
    base = _eth_base_name(eth_npy_file)
    candidates = [
        os.path.join(data_dir, "replica_eth_smoke", f"{base}_replica_packets.csv"),
        os.path.join(data_dir, f"{base}_replica_packets.csv"),
        os.path.join(data_dir, f"{base}_preprocessed.csv"),
        os.path.join(data_dir, f"{base}.csv"),
    ]
    existing = [candidate for candidate in candidates if os.path.exists(candidate)]
    labeled = []
    for candidate in existing:
        try:
            with open(candidate, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
        except Exception:
            header = []
        if "Label" in header:
            labeled.append(candidate)
    for candidate in labeled:
        return candidate
    for candidate in existing:
        return candidate
    return None
