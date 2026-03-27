"""
setup_datasets.py — Download and prepare datasets for Research_Auto_IDS.

This script automates the dataset acquisition and feature engineering
pipeline so that researchers can reproduce results from scratch.

Usage:
    python setup_datasets.py              # Full setup (download + engineer)
    python setup_datasets.py --skip-download  # Skip downloads, run engineering only
    python setup_datasets.py --dry-run    # Show what would be done

Phase 1 (DPI): PCAP extraction now includes Ethernet/IP/UDP/TCP header
fields and the first 16 application-layer payload bytes plus Shannon
entropy, enabling payload-sensitive intrusion detection.
"""

import argparse
import csv
import json
import os
import sys
import shutil
import hashlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
ENGINEERED_DIR = os.path.join(DATASETS_DIR, "replica_can_b1_engineered")
ETH_SMOKE_DIR = os.path.join(DATASETS_DIR, "replica_eth_smoke")
ETH_PROVENANCE_COLUMNS = [
    "session_id",
    "attack_type",
    "label_source",
    "label_granularity",
]
DEFAULT_ETH_LABEL_MANIFEST_CANDIDATES = [
    os.path.join(DATASETS_DIR, "autoeth-intrusion-dataset", "eth_label_manifest.json"),
    os.path.join(BASE_DIR, "data", "manifests", "autoeth_label_manifest.json"),
]


# ── Dataset sources ──────────────────────────────────────────────────────────
# Researchers must download datasets manually from these sources and place
# them under datasets/ as described below.
DATASET_SOURCES = {
    "Car-Hacking Dataset": {
        "url": "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset",
        "description": "CAN bus attack dataset (DoS, Fuzzy, Gear, RPM spoofing)",
        "target_dir": os.path.join(DATASETS_DIR, "Car-Hacking Dataset"),
        "expected_files": [
            "DoS_dataset.csv",
            "Fuzzy_dataset.csv",
            "gear_dataset.csv",
            "RPM_dataset.csv",
        ],
    },
    "AutoETH Intrusion Dataset": {
        "url": "https://zenodo.org/records/14643663",
        "description": "Automotive Ethernet intrusion dataset with PCAP captures",
        "target_dir": os.path.join(DATASETS_DIR, "autoeth-intrusion-dataset"),
        "expected_files": [
            "driving_01_injected.pcap",
            "driving_01_original.pcap",
            "driving_02_injected.pcap",
            "driving_02_original.pcap",
            "indoors_01_injected.pcap",
            "indoors_01_original.pcap",
            "indoors_02_injected.pcap",
            "indoors_02_original.pcap",
        ],
    },
}


def print_download_instructions():
    """Print manual download instructions for required datasets."""
    print("=" * 72)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 72)
    print()
    print("This research uses two publicly available datasets that must be")
    print("downloaded manually due to licensing and size constraints.")
    print()

    for name, info in DATASET_SOURCES.items():
        print(f"  {name}")
        print(f"  Description : {info['description']}")
        print(f"  Download URL: {info['url']}")
        print(f"  Place into  : {info['target_dir']}")
        print(f"  Files needed: {', '.join(info['expected_files'])}")
        print()

    print("After downloading, place the files in the paths shown above,")
    print("then re-run this script to prepare engineered features.")
    print("=" * 72)


def check_datasets_present() -> dict:
    """Check which datasets are already downloaded."""
    status = {}
    for name, info in DATASET_SOURCES.items():
        missing = []
        for f in info["expected_files"]:
            path = os.path.join(info["target_dir"], f)
            if not os.path.exists(path):
                missing.append(f)
        status[name] = {
            "present": len(missing) == 0,
            "missing": missing,
            "dir_exists": os.path.isdir(info["target_dir"]),
        }
    return status


def prepare_can_training_csvs():
    """
    Extract CAN training CSVs from the Car-Hacking Dataset.

    The raw Car-Hacking Dataset files have columns:
        Timestamp, CAN_ID, DLC, D0, D1, D2, D3, D4, D5, D6, D7, Flag

    We create smaller per-attack training CSVs under datasets/.
    """
    import pandas as pd

    source_dir = os.path.join(DATASETS_DIR, "Car-Hacking Dataset")
    mapping = {
        "DoS_dataset.csv": "can_dos_train.csv",
        "Fuzzy_dataset.csv": "can_fuzzy_train.csv",
        "gear_dataset.csv": "can_gear_train.csv",
        "RPM_dataset.csv": "can_rpm_train.csv",
    }

    for src_name, dst_name in mapping.items():
        src_path = os.path.join(source_dir, src_name)
        dst_path = os.path.join(DATASETS_DIR, dst_name)

        if os.path.exists(dst_path):
            print(f"  [skip] {dst_name} already exists")
            continue

        if not os.path.exists(src_path):
            print(f"  [warn] Source not found: {src_path}")
            continue

        print(f"  Creating {dst_name} from {src_name}...")
        df = pd.read_csv(src_path, header=None)
        df.columns = [
            "Timestamp", "CAN_ID", "DLC",
            "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
            "Flag",
        ]
        df.to_csv(dst_path, index=False)
        print(f"    -> {len(df):,} rows written")

    # Normal data — try from normal_run_data
    normal_dst = os.path.join(DATASETS_DIR, "can_normal_train.csv")
    if not os.path.exists(normal_dst):
        normal_src = os.path.join(source_dir, "normal_run_data", "normal_run_data.txt")
        if os.path.exists(normal_src):
            print(f"  Creating can_normal_train.csv from normal_run_data.txt...")
            df = pd.read_csv(normal_src, header=None)
            df.columns = [
                "Timestamp", "CAN_ID", "DLC",
                "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
                "Flag",
            ]
            df.to_csv(normal_dst, index=False)
            print(f"    -> {len(df):,} rows written")


def run_can_feature_engineering():
    """
    Engineer 16 CAN features from raw training CSVs.

    Produces datasets/replica_can_b1_engineered/*.csv
    """
    import pandas as pd
    sys.path.insert(0, os.path.join(BASE_DIR, "src_replica"))
    from features_can_replica import add_can_engineered_features

    os.makedirs(ENGINEERED_DIR, exist_ok=True)

    can_files = [
        "can_dos_train.csv",
        "can_fuzzy_train.csv",
        "can_gear_train.csv",
        "can_rpm_train.csv",
    ]

    for fname in can_files:
        src_path = os.path.join(DATASETS_DIR, fname)
        dst_path = os.path.join(ENGINEERED_DIR, fname)

        if os.path.exists(dst_path):
            print(f"  [skip] {fname} already engineered")
            continue

        if not os.path.exists(src_path):
            print(f"  [warn] Source not found: {src_path}")
            continue

        print(f"  Engineering features for {fname}...")
        df = pd.read_csv(src_path)
        df_eng = add_can_engineered_features(df, window=200)
        df_eng.to_csv(dst_path, index=False)
        print(f"    -> {len(df_eng):,} rows with 16 features")


def _resolve_eth_label_manifest_path(manifest_path: str | None) -> str | None:
    candidates = []
    if manifest_path:
        candidates.append(manifest_path if os.path.isabs(manifest_path) else os.path.join(BASE_DIR, manifest_path))
    candidates.extend(DEFAULT_ETH_LABEL_MANIFEST_CANDIDATES)
    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(candidate):
            return candidate
    return None


def load_eth_label_manifest(manifest_path: str | None) -> dict:
    resolved = _resolve_eth_label_manifest_path(manifest_path)
    if not resolved:
        return {}
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    sessions = {}
    for session in data.get("sessions", []):
        scenario = str(session.get("scenario") or session.get("session_id") or "").strip()
        if not scenario:
            continue
        sessions[scenario] = session
    return sessions


def label_eth_packet_from_manifest(
    scenario: str,
    timestamp_sec: float,
    manifest_sessions: dict,
) -> dict:
    session = manifest_sessions.get(scenario)
    if not session:
        label = 1 if ("injected" in scenario or "attack" in scenario) else 0
        return {
            "Label": label,
            "session_id": f"autoeth::{scenario}",
            "attack_type": "avtp_injection" if label == 1 else "benign",
            "label_source": "scenario_placeholder",
            "label_granularity": "scenario",
        }

    session_id = str(session.get("session_id") or f"autoeth::{scenario}")
    default_label = int(session.get("default_label", 0))
    default_attack_type = str(session.get("default_attack_type") or ("benign" if default_label == 0 else "unknown_attack"))
    default_label_source = str(session.get("default_label_source") or "session_default")
    default_label_granularity = str(session.get("default_label_granularity") or "session")

    for interval in session.get("intervals", []):
        start_ts = interval.get("start_ts")
        end_ts = interval.get("end_ts")
        if start_ts is None:
            continue
        start_ts = float(start_ts)
        end_ok = True if end_ts is None else float(timestamp_sec) < float(end_ts)
        if float(timestamp_sec) >= start_ts and end_ok:
            label = int(interval.get("label", 1))
            return {
                "Label": label,
                "session_id": session_id,
                "attack_type": str(interval.get("attack_type") or default_attack_type),
                "label_source": str(interval.get("label_source") or "packet_ground_truth"),
                "label_granularity": str(interval.get("label_granularity") or "packet"),
            }

    return {
        "Label": default_label,
        "session_id": session_id,
        "attack_type": default_attack_type,
        "label_source": default_label_source,
        "label_granularity": default_label_granularity,
    }


def bootstrap_eth_label_manifest(
    output_path: str,
    packet_csv_dir: str | None = None,
) -> str:
    import pandas as pd

    packet_csv_dir = packet_csv_dir or ETH_SMOKE_DIR
    sessions = []
    if not os.path.isdir(packet_csv_dir):
        raise FileNotFoundError(f"ETH packet CSV directory not found: {packet_csv_dir}")

    for name in sorted(os.listdir(packet_csv_dir)):
        if not name.startswith("eth_") or not name.endswith("_replica_packets.csv"):
            continue
        csv_path = os.path.join(packet_csv_dir, name)
        try:
            df = pd.read_csv(csv_path, usecols=["timestamp_sec", "timestamp_usec"])
        except Exception:
            continue
        if df.empty:
            continue
        ts = pd.to_numeric(df["timestamp_sec"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        usec = pd.to_numeric(df["timestamp_usec"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        timestamp = ts + usec / 1_000_000.0
        scenario = name[len("eth_"):-len("_replica_packets.csv")]
        is_injected = ("injected" in scenario) or ("attack" in scenario)
        session = {
            "scenario": scenario,
            "session_id": f"autoeth::{scenario}",
            "observed_start_ts": float(timestamp.min()),
            "observed_end_ts": float(timestamp.max()),
            "default_label": 0,
            "default_attack_type": "benign",
            "default_label_source": "manual_pending",
            "default_label_granularity": "packet",
            "annotation_status": "needs_review",
            "intervals": [],
        }
        if is_injected:
            session["intervals"].append(
                {
                    "start_ts": float(timestamp.min()),
                    "end_ts": float(timestamp.max()) + 1e-9,
                    "label": 1,
                    "attack_type": "avtp_injection",
                    "label_source": "inferred_full_session",
                    "label_granularity": "session",
                    "annotation_status": "needs_review",
                }
            )
        sessions.append(session)

    manifest = {
        "version": 1,
        "description": "Bootstrapped AutoETH manifest from packet CSV time bounds. Intervals marked inferred_full_session are not research-grade ground truth and require review.",
        "sessions": sessions,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return output_path


def _scenario_from_eth_packet_csv_name(name: str) -> str | None:
    prefix = "eth_"
    suffix = "_replica_packets.csv"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix):-len(suffix)]


def apply_eth_label_manifest_to_packet_csvs(
    manifest_path: str,
    packet_csv_dir: str | None = None,
) -> int:
    import pandas as pd

    packet_csv_dir = packet_csv_dir or ETH_SMOKE_DIR
    manifest_sessions = load_eth_label_manifest(manifest_path)
    if not manifest_sessions:
        raise ValueError(f"No sessions found in ETH label manifest: {manifest_path}")
    if not os.path.isdir(packet_csv_dir):
        raise FileNotFoundError(f"ETH packet CSV directory not found: {packet_csv_dir}")

    updated = 0
    for name in sorted(os.listdir(packet_csv_dir)):
        scenario = _scenario_from_eth_packet_csv_name(name)
        if not scenario:
            continue
        csv_path = os.path.join(packet_csv_dir, name)
        df = pd.read_csv(csv_path)
        if "timestamp_sec" not in df.columns:
            continue
        sec = pd.to_numeric(df["timestamp_sec"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        usec = pd.to_numeric(df.get("timestamp_usec", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        timestamp = sec + usec / 1_000_000.0
        labels = []
        session_ids = []
        attack_types = []
        label_sources = []
        label_granularities = []
        for ts in timestamp.tolist():
            info = label_eth_packet_from_manifest(scenario, ts, manifest_sessions)
            labels.append(int(info["Label"]))
            session_ids.append(str(info["session_id"]))
            attack_types.append(str(info["attack_type"]))
            label_sources.append(str(info["label_source"]))
            label_granularities.append(str(info["label_granularity"]))
        df["Label"] = labels
        df["session_id"] = session_ids
        df["attack_type"] = attack_types
        df["label_source"] = label_sources
        df["label_granularity"] = label_granularities
        df.to_csv(csv_path, index=False)
        updated += 1
    return updated


# ── DPI helper ────────────────────────────────────────────────────────────────
_ETH_PAYLOAD_BYTES = 16  # Number of application-layer bytes to capture
_ETH_PAYLOAD_COLS = [f"payload_b{i}" for i in range(_ETH_PAYLOAD_BYTES)]


def _extract_dpi_fields(pkt) -> dict:
    """
    Extract Deep Packet Inspection (DPI) fields from a scapy packet.

    Returns a dict with:
        eth_type       : Ethernet EtherType (0–65535, normalised /65535)
        ip_proto       : IP protocol number (0–255, normalised /255)
        src_port       : TCP/UDP source port (0–65535, normalised /65535)
        dst_port       : TCP/UDP destination port (0–65535, normalised /65535)
        payload_len    : raw application-layer payload length in bytes
        payload_entropy: Shannon entropy of first 64 payload bytes (0.0–1.0)
        payload_b0 … payload_b15: first 16 payload bytes normalised to [0,1]
    """
    import math

    result: dict = {
        "eth_type": 0.0,
        "ip_proto": 0.0,
        "src_port": 0.0,
        "dst_port": 0.0,
        "payload_len": 0,
        "payload_entropy": 0.0,
    }
    for col in _ETH_PAYLOAD_COLS:
        result[col] = 0.0

    try:
        from scapy.layers.l2 import Ether
        from scapy.layers.inet import IP, TCP, UDP

        # Ethernet EtherType
        if pkt.haslayer(Ether):
            result["eth_type"] = float(pkt[Ether].type) / 65535.0

        # IP layer
        if pkt.haslayer(IP):
            result["ip_proto"] = float(pkt[IP].proto) / 255.0

        # Transport layer ports
        if pkt.haslayer(TCP):
            result["src_port"] = float(pkt[TCP].sport) / 65535.0
            result["dst_port"] = float(pkt[TCP].dport) / 65535.0
            app_payload = bytes(pkt[TCP].payload)
        elif pkt.haslayer(UDP):
            result["src_port"] = float(pkt[UDP].sport) / 65535.0
            result["dst_port"] = float(pkt[UDP].dport) / 65535.0
            app_payload = bytes(pkt[UDP].payload)
        else:
            # Fallback: everything after IP header
            if pkt.haslayer(IP):
                app_payload = bytes(pkt[IP].payload)
            else:
                from scapy.layers.l2 import Ether
                app_payload = bytes(pkt.payload) if pkt.haslayer(Ether) else b""

        result["payload_len"] = len(app_payload)

        # Payload byte features (first _ETH_PAYLOAD_BYTES bytes, normalised)
        for i in range(_ETH_PAYLOAD_BYTES):
            result[f"payload_b{i}"] = float(app_payload[i]) / 255.0 if i < len(app_payload) else 0.0

        # Shannon entropy over first 64 bytes
        sample = app_payload[:64]
        if sample:
            from collections import Counter
            counts = Counter(sample)
            total = len(sample)
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in counts.values() if c > 0
            )
            max_entropy = math.log2(min(total, 256)) if total > 1 else 1.0
            result["payload_entropy"] = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    except Exception:
        pass  # Non-IP or malformed packets: all fields remain 0

    return result


def prepare_eth_preprocessed(manifest_path: str | None = None):
    """
    Extract Ethernet packet CSVs from PCAP files and create image .npy arrays.

    Requires: scapy (pip install scapy)
    Creates: datasets/replica_eth_smoke/*.csv and datasets/eth_*_images*.npy
    """
    try:
        from scapy.all import rdpcap
    except ImportError:
        print("  [warn] scapy not installed — skipping Ethernet PCAP extraction.")
        print("         Install with: pip install scapy")
        print("         Or provide pre-extracted CSVs in datasets/replica_eth_smoke/")
        return

    import numpy as np
    import pandas as pd

    pcap_dir = os.path.join(DATASETS_DIR, "autoeth-intrusion-dataset")
    os.makedirs(ETH_SMOKE_DIR, exist_ok=True)
    manifest_sessions = load_eth_label_manifest(manifest_path)
    resolved_manifest = _resolve_eth_label_manifest_path(manifest_path)
    if resolved_manifest:
        print(f"  Using ETH label manifest: {resolved_manifest}")
    else:
        print("  No ETH label manifest found; falling back to scenario-level placeholder labels.")

    pcap_files = [
        "driving_01_injected",
        "driving_01_original",
        "driving_02_injected",
        "driving_02_original",
        "indoors_01_injected",
        "indoors_01_original",
        "indoors_02_injected",
        "indoors_02_original",
    ]

    for scenario in pcap_files:
        csv_dst = os.path.join(ETH_SMOKE_DIR, f"eth_{scenario}_replica_packets.csv")
        if os.path.exists(csv_dst):
            try:
                with open(csv_dst, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
            except Exception:
                header = []
            # Require label provenance AND new DPI payload columns
            has_required = (
                "Label" in header
                and all(col in header for col in ETH_PROVENANCE_COLUMNS)
                and "payload_entropy" in header
                and "payload_b0" in header
            )
            if has_required:
                print(f"  [skip] eth_{scenario}_replica_packets.csv exists with label provenance + payload columns")
                continue
            print(f"  [refresh] eth_{scenario}_replica_packets.csv missing columns; regenerating")

        pcap_path = os.path.join(pcap_dir, f"{scenario}.pcap")
        if not os.path.exists(pcap_path):
            print(f"  [warn] PCAP not found: {pcap_path}")
            continue

        print(f"  Extracting {scenario}.pcap -> CSV (DPI enabled)...")
        packets = rdpcap(pcap_path)
        rows = []
        for pkt in packets:
            ts = float(pkt.time)
            ts_sec = int(ts)
            ts_usec = int((ts - ts_sec) * 1_000_000)
            raw = bytes(pkt)
            label_info = label_eth_packet_from_manifest(
                scenario=scenario,
                timestamp_sec=ts,
                manifest_sessions=manifest_sessions,
            )

            # ── Deep Packet Inspection ─────────────────────────────────────
            dpi = _extract_dpi_fields(pkt)
            rows.append({
                "timestamp_sec": ts_sec,
                "timestamp_usec": ts_usec,
                "captured_len": len(raw),
                "original_len": len(raw),
                **dpi,
                **label_info,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_dst, index=False)
        print(f"    -> {len(df):,} packets extracted (with DPI payload features)")


def verify_setup():
    """Verify all required files exist for training and evaluation."""
    print()
    print("VERIFICATION")
    print("-" * 40)

    checks = [
        ("CAN DoS training", os.path.join(DATASETS_DIR, "can_dos_train.csv")),
        ("CAN Fuzzy training", os.path.join(DATASETS_DIR, "can_fuzzy_train.csv")),
        ("CAN Gear training", os.path.join(DATASETS_DIR, "can_gear_train.csv")),
        ("CAN RPM training", os.path.join(DATASETS_DIR, "can_rpm_train.csv")),
        ("CAN Normal training", os.path.join(DATASETS_DIR, "can_normal_train.csv")),
        ("Engineered CAN DoS", os.path.join(ENGINEERED_DIR, "can_dos_train.csv")),
        ("Engineered CAN Fuzzy", os.path.join(ENGINEERED_DIR, "can_fuzzy_train.csv")),
        ("Engineered CAN Gear", os.path.join(ENGINEERED_DIR, "can_gear_train.csv")),
        ("Engineered CAN RPM", os.path.join(ENGINEERED_DIR, "can_rpm_train.csv")),
        ("ETH smoke packets", os.path.join(ETH_SMOKE_DIR, "eth_driving_01_injected_replica_packets.csv")),
        ("Split V1", os.path.join(BASE_DIR, "data", "splits", "split_v1.json")),
        ("Split V2", os.path.join(BASE_DIR, "data", "splits", "split_v2_domain_balanced.json")),
        ("Deployment config", os.path.join(BASE_DIR, "configs", "deployment.example.json")),
    ]

    all_ok = True
    for label, path in checks:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status:7s}] {label}")
        if not exists:
            all_ok = False

    eth_smoke_csv = os.path.join(ETH_SMOKE_DIR, "eth_driving_01_injected_replica_packets.csv")
    if os.path.exists(eth_smoke_csv):
        try:
            with open(eth_smoke_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
        except Exception:
            header = []
        has_required = "Label" in header and all(col in header for col in ETH_PROVENANCE_COLUMNS)
        print(f"  [{'OK' if has_required else 'WARN':7s}] ETH smoke label provenance columns")
        if not has_required:
            print("           Existing Ethernet replay CSVs are missing label provenance; rerun setup to regenerate them.")
            all_ok = False

    print()
    if all_ok:
        print("All required files are present. Ready for training!")
    else:
        print("Some files are missing. See instructions above.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for Research_Auto_IDS reproducibility."
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download instructions; only run feature engineering."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing."
    )
    parser.add_argument(
        "--eth-label-manifest", type=str, default=None,
        help="Optional JSON manifest with timestamp-based ETH labels."
    )
    parser.add_argument(
        "--bootstrap-eth-label-manifest-out", type=str, default=None,
        help="Write a draft AutoETH label manifest from existing ETH packet CSVs."
    )
    parser.add_argument(
        "--apply-eth-label-manifest", type=str, default=None,
        help="Apply an ETH label manifest to existing ETH packet CSVs in replica_eth_smoke."
    )
    args = parser.parse_args()

    print("Research_Auto_IDS — Dataset Setup")
    print("=" * 40)
    print()

    # Step 1: Check existing datasets
    print("[Step 1] Checking existing datasets...")
    status = check_datasets_present()
    all_present = all(s["present"] for s in status.values())

    for name, s in status.items():
        if s["present"]:
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name} — missing: {', '.join(s['missing'])}")

    if not all_present and not args.skip_download:
        print()
        print_download_instructions()
        print()
        resp = input("Have you downloaded the datasets? [y/N]: ").strip().lower()
        if resp != "y":
            print("Please download the datasets first, then re-run this script.")
            return 1

        # Re-check
        status = check_datasets_present()
        all_present = all(s["present"] for s in status.values())
        if not all_present:
            print("Datasets still missing. Please check the paths above.")
            return 1

    if args.dry_run:
        print()
        print("[DRY RUN] Would perform:")
        print("  1. Extract CAN training CSVs from Car-Hacking Dataset")
        print("  2. Engineer 16 CAN features -> replica_can_b1_engineered/")
        print("  3. Extract ETH packet CSVs from PCAP files (requires scapy)")
        print("  4. Verify all files present")
        return 0

    if args.bootstrap_eth_label_manifest_out:
        out_path = (
            args.bootstrap_eth_label_manifest_out
            if os.path.isabs(args.bootstrap_eth_label_manifest_out)
            else os.path.join(BASE_DIR, args.bootstrap_eth_label_manifest_out)
        )
        written = bootstrap_eth_label_manifest(out_path)
        print(f"Bootstrapped ETH label manifest to {written}")
    if args.apply_eth_label_manifest:
        manifest_path = (
            args.apply_eth_label_manifest
            if os.path.isabs(args.apply_eth_label_manifest)
            else os.path.join(BASE_DIR, args.apply_eth_label_manifest)
        )
        updated = apply_eth_label_manifest_to_packet_csvs(manifest_path)
        print(f"Applied ETH label manifest to {updated} packet CSVs")

    # Step 2: Prepare CAN training CSVs
    print()
    print("[Step 2] Preparing CAN training CSVs...")
    prepare_can_training_csvs()

    # Step 3: CAN feature engineering
    print()
    print("[Step 3] Running CAN feature engineering (16 features)...")
    run_can_feature_engineering()

    # Step 4: ETH PCAP extraction
    print()
    print("[Step 4] Preparing Ethernet packet data...")
    prepare_eth_preprocessed(manifest_path=args.eth_label_manifest)

    # Step 5: Verification
    print()
    print("[Step 5] Verifying setup...")
    ok = verify_setup()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
