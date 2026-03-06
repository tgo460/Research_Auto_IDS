"""
setup_datasets.py — Download and prepare datasets for Research_Auto_IDS.

This script automates the dataset acquisition and feature engineering
pipeline so that researchers can reproduce results from scratch.

Usage:
    python setup_datasets.py              # Full setup (download + engineer)
    python setup_datasets.py --skip-download  # Skip downloads, run engineering only
    python setup_datasets.py --dry-run    # Show what would be done
"""

import argparse
import os
import sys
import shutil
import hashlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
ENGINEERED_DIR = os.path.join(DATASETS_DIR, "replica_can_b1_engineered")
ETH_SMOKE_DIR = os.path.join(DATASETS_DIR, "replica_eth_smoke")


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


def prepare_eth_preprocessed():
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
            print(f"  [skip] eth_{scenario}_replica_packets.csv exists")
            continue

        pcap_path = os.path.join(pcap_dir, f"{scenario}.pcap")
        if not os.path.exists(pcap_path):
            print(f"  [warn] PCAP not found: {pcap_path}")
            continue

        print(f"  Extracting {scenario}.pcap -> CSV...")
        packets = rdpcap(pcap_path)
        rows = []
        for pkt in packets:
            ts = float(pkt.time)
            ts_sec = int(ts)
            ts_usec = int((ts - ts_sec) * 1_000_000)
            raw = bytes(pkt)
            rows.append({
                "timestamp_sec": ts_sec,
                "timestamp_usec": ts_usec,
                "captured_len": len(raw),
                "original_len": len(raw),
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_dst, index=False)
        print(f"    -> {len(df):,} packets extracted")


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
    prepare_eth_preprocessed()

    # Step 5: Verification
    print()
    print("[Step 5] Verifying setup...")
    ok = verify_setup()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
