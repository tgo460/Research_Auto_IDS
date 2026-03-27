import json
from collections import deque

import numpy as np
import pandas as pd
import pytest

from setup_datasets import (
    apply_eth_label_manifest_to_packet_csvs,
    bootstrap_eth_label_manifest,
    label_eth_packet_from_manifest,
    load_eth_label_manifest,
)
from evaluate import _validate_split_manifest
from src_replica.cascade_eval_replica import _plan_pairs
from src_replica.correlation_replica import correlate_can_eth
from src_replica.dataloader_correlated_replica import CorrelatedHybridVehicleDataset, _load_eth_labels
from src_replica.data_resolvers import resolve_eth_packet_csv
from src_replica.features_can_replica import add_can_engineered_features
from src_replica.preprocessing_standard import (
    STANDARD_CAN_FEATURES_16,
    build_eth_image_windows,
    standardize_can_dataframe,
    standardize_eth_packet_dataframe,
)
from src_replica.runtime.adapters import CsvCanIngest, CsvEthIngest, to_can_window


ETH_PROVENANCE_HEADER = "session_id,attack_type,label_source,label_granularity"
ETH_PROVENANCE_ROW = "sess_demo,avtp_injection,packet_ground_truth,packet"


def test_csv_eth_ingest_requires_label_column(tmp_path):
    csv_path = tmp_path / "eth_demo_injected_replica_packets.csv"
    csv_path.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Label"):
        CsvEthIngest(str(csv_path))


def test_csv_eth_ingest_requires_label_provenance_columns(tmp_path):
    csv_path = tmp_path / "eth_demo_injected_replica_packets.csv"
    csv_path.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len,Label\n"
        "0,1,0,64,64,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="session_id"):
        CsvEthIngest(str(csv_path))


def test_csv_eth_ingest_uses_explicit_label_column(tmp_path):
    csv_path = tmp_path / "eth_demo_replica_packets.csv"
    csv_path.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len,Label,"
        + ETH_PROVENANCE_HEADER + "\n"
        "0,1,0,64,64,1," + ETH_PROVENANCE_ROW + "\n",
        encoding="utf-8",
    )
    ingest = CsvEthIngest(str(csv_path))
    frame = ingest.read_frame()
    assert frame is not None
    assert frame["label"] == 1
    assert frame["session_id"] == "sess_demo"


def test_csv_can_ingest_respects_start_row(tmp_path):
    csv_path = tmp_path / "can_demo.csv"
    csv_path.write_text(
        "Timestamp,CAN_ID,DLC,D0,D1,D2,D3,D4,D5,D6,D7,Label\n"
        "0.0,0.1,1,0,0,0,0,0,0,0,0,0\n"
        "0.1,0.2,1,0,0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    ingest = CsvCanIngest(str(csv_path), start_row=1)
    frame = ingest.read_frame()
    assert frame is not None
    assert frame["label"] == 1


def test_to_can_window_computes_engineered_features_when_missing():
    frames = deque()
    for idx in range(100):
        frames.append(
            {
                "timestamp": float(idx) * 0.001,
                "can_id": 0.1 if idx % 2 == 0 else 0.2,
                "dlc": 1.0,
                "payload": [float((idx + j) % 8) / 7.0 for j in range(8)],
                "label": 0,
            }
        )

    mat = to_can_window(frames, expected_size=100)
    assert mat.shape == (100, 16)
    assert mat[:, 10:].sum() > 0.0


def test_validate_split_manifest_flags_overlap(tmp_path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "a.csv").write_text("x\n", encoding="utf-8")

    manifest = {
        "modalities": {
            "can": {
                "train": ["a.csv"],
                "val": ["a.csv"],
                "test": [],
            },
            "eth": {
                "train": [],
                "val": [],
                "test": [],
            },
        }
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _validate_split_manifest(str(manifest_path), str(datasets_dir))
    assert report["ok"] is False
    assert any("overlap" in err for err in report["errors"])


def test_validate_split_manifest_accepts_disjoint_row_ranges(tmp_path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "can_normal_train.csv").write_text("x\n", encoding="utf-8")

    manifest = {
        "modalities": {
            "can": {
                "train": [{"path": "can_normal_train.csv", "row_start": 0, "row_stop": 10}],
                "val": [{"path": "can_normal_train.csv", "row_start": 10, "row_stop": 20}],
                "test": [{"path": "can_normal_train.csv", "row_start": 20, "row_stop": 30}],
            },
            "eth": {"train": [], "val": [], "test": []},
        }
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _validate_split_manifest(str(manifest_path), str(datasets_dir))
    assert report["ok"] is True
    assert report["errors"] == []


def test_load_eth_label_manifest_and_interval_labeling(tmp_path):
    manifest_path = tmp_path / "eth_label_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "scenario": "demo_injected",
                        "session_id": "sess_demo",
                        "default_label": 0,
                        "default_attack_type": "benign",
                        "default_label_source": "packet_ground_truth",
                        "default_label_granularity": "packet",
                        "intervals": [
                            {
                                "start_ts": 10.0,
                                "end_ts": 12.0,
                                "label": 1,
                                "attack_type": "avtp_injection",
                                "label_source": "packet_ground_truth",
                                "label_granularity": "packet",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = load_eth_label_manifest(str(manifest_path))
    benign = label_eth_packet_from_manifest("demo_injected", 9.5, manifest)
    attack = label_eth_packet_from_manifest("demo_injected", 10.5, manifest)
    fallback = label_eth_packet_from_manifest("unknown_session", 10.5, manifest)

    assert benign["Label"] == 0
    assert benign["label_source"] == "packet_ground_truth"
    assert attack["Label"] == 1
    assert attack["attack_type"] == "avtp_injection"
    assert fallback["label_source"] == "scenario_placeholder"


def test_bootstrap_eth_label_manifest_from_packet_csvs(tmp_path):
    eth_dir = tmp_path / "replica_eth_smoke"
    eth_dir.mkdir(parents=True)
    (eth_dir / "eth_driving_01_injected_replica_packets.csv").write_text(
        "timestamp_sec,timestamp_usec,captured_len,original_len,Label,session_id,attack_type,label_source,label_granularity\n"
        "10,0,64,64,1,s1,avtp_injection,packet_ground_truth,packet\n"
        "12,500000,64,64,1,s1,avtp_injection,packet_ground_truth,packet\n",
        encoding="utf-8",
    )
    (eth_dir / "eth_driving_01_original_replica_packets.csv").write_text(
        "timestamp_sec,timestamp_usec,captured_len,original_len,Label,session_id,attack_type,label_source,label_granularity\n"
        "20,0,64,64,0,s2,benign,packet_ground_truth,packet\n"
        "21,0,64,64,0,s2,benign,packet_ground_truth,packet\n",
        encoding="utf-8",
    )

    out = tmp_path / "autoeth_label_manifest.json"
    written = bootstrap_eth_label_manifest(str(out), packet_csv_dir=str(eth_dir))
    assert written == str(out)
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert len(manifest["sessions"]) == 2
    injected = next(s for s in manifest["sessions"] if s["scenario"] == "driving_01_injected")
    assert injected["intervals"]
    assert injected["intervals"][0]["label_source"] == "inferred_full_session"


def test_apply_eth_label_manifest_to_packet_csvs(tmp_path):
    eth_dir = tmp_path / "replica_eth_smoke"
    eth_dir.mkdir(parents=True)
    csv_path = eth_dir / "eth_driving_01_injected_replica_packets.csv"
    csv_path.write_text(
        "timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "10,0,64,64\n"
        "11,0,64,64\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "scenario": "driving_01_injected",
                        "session_id": "sess_demo",
                        "default_label": 0,
                        "default_attack_type": "benign",
                        "default_label_source": "manual_pending",
                        "default_label_granularity": "packet",
                        "intervals": [
                            {
                                "start_ts": 10.5,
                                "end_ts": 11.5,
                                "label": 1,
                                "attack_type": "avtp_injection",
                                "label_source": "attack_log_interval",
                                "label_granularity": "packet",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    updated = apply_eth_label_manifest_to_packet_csvs(str(manifest_path), packet_csv_dir=str(eth_dir))
    assert updated == 1
    relabeled = pd.read_csv(csv_path)
    assert relabeled["Label"].tolist() == [0, 1]
    assert relabeled["label_source"].tolist() == ["manual_pending", "attack_log_interval"]


def test_add_can_engineered_features_preserves_normalized_can_id_distinctions():
    df = pd.DataFrame(
        {
            "Timestamp": [0.0, 0.001, 0.002],
            "CAN_ID": [0.10, 0.20, 0.10],
            "DLC": [1.0, 1.0, 1.0],
            "Label": [0, 0, 1],
            **{f"D{i}": [0.0, 0.5, 1.0] for i in range(8)},
        }
    )

    engineered = add_can_engineered_features(df, window=2)
    assert "can_id_freq_global" in engineered.columns
    assert engineered.loc[0, "can_id_freq_global"] == pytest.approx(1.0)
    assert engineered.loc[1, "can_id_freq_global"] == pytest.approx(0.5)
    assert engineered.loc[2, "can_id_freq_global"] == pytest.approx(2.0 / 3.0, rel=1e-5)
    assert engineered.loc[2, "can_id_freq_global"] > engineered.loc[1, "can_id_freq_global"]


def test_load_eth_labels_prefers_label_csv_when_packet_csv_is_unlabeled(tmp_path):
    datasets_dir = tmp_path / "datasets"
    smoke_dir = datasets_dir / "replica_eth_smoke"
    smoke_dir.mkdir(parents=True)

    packet_csv = smoke_dir / "eth_demo_original_replica_packets.csv"
    packet_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n",
        encoding="utf-8",
    )
    label_csv = datasets_dir / "eth_demo_original.csv"
    label_csv.write_text(
        "Label,session_id,attack_type,label_source,label_granularity\n"
        "1,sess_demo,avtp_injection,packet_ground_truth,packet\n",
        encoding="utf-8",
    )
    npy_path = datasets_dir / "eth_demo_original_images.npy"
    np.save(str(npy_path), np.zeros((1, 32, 32), dtype=np.uint8))

    labels, source, provenance = _load_eth_labels(str(packet_csv), str(npy_path))
    assert labels.tolist() == [1]
    assert source.endswith("eth_demo_original.csv")
    assert provenance["label_source"] == "packet_ground_truth"


def test_load_eth_labels_prefers_canonical_csv_over_preprocessed_csv(tmp_path):
    datasets_dir = tmp_path / "datasets"
    smoke_dir = datasets_dir / "replica_eth_smoke"
    smoke_dir.mkdir(parents=True)

    packet_csv = smoke_dir / "eth_demo_original_replica_packets.csv"
    packet_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n",
        encoding="utf-8",
    )
    canonical_csv = datasets_dir / "eth_demo_original.csv"
    canonical_csv.write_text(
        "Label,session_id,attack_type,label_source,label_granularity\n"
        "0,sess_demo,benign,packet_ground_truth,packet\n",
        encoding="utf-8",
    )
    preprocessed_csv = datasets_dir / "eth_demo_original_preprocessed.csv"
    preprocessed_csv.write_text(
        "Label,session_id,attack_type,label_source,label_granularity\n"
        "1,sess_demo,avtp_injection,packet_ground_truth,packet\n",
        encoding="utf-8",
    )
    npy_path = datasets_dir / "eth_demo_original_images.npy"
    np.save(str(npy_path), np.zeros((1, 32, 32), dtype=np.uint8))

    labels, source, provenance = _load_eth_labels(str(packet_csv), str(npy_path))
    assert labels.tolist() == [0]
    assert source.endswith("eth_demo_original.csv")
    assert provenance["attack_type"] == "benign"


def test_load_eth_labels_requires_explicit_label_source(tmp_path):
    packet_csv = tmp_path / "eth_demo_replica_packets.csv"
    packet_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n",
        encoding="utf-8",
    )
    npy_path = tmp_path / "eth_demo_images.npy"
    np.save(str(npy_path), np.zeros((1, 32, 32), dtype=np.uint8))

    with pytest.raises(ValueError, match="ETH labels with provenance are required"):
        _load_eth_labels(str(packet_csv), str(npy_path))


def test_resolve_eth_packet_csv_prefers_labeled_candidate(tmp_path):
    data_dir = tmp_path / "datasets"
    smoke_dir = data_dir / "replica_eth_smoke"
    smoke_dir.mkdir(parents=True)

    unlabeled = smoke_dir / "eth_demo_original_replica_packets.csv"
    unlabeled.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n",
        encoding="utf-8",
    )
    labeled = data_dir / "eth_demo_original.csv"
    labeled.write_text(
        "timestamp_sec,timestamp_usec,captured_len,original_len,Label,session_id,attack_type,label_source,label_granularity\n"
        "1,0,64,64,0,sess_demo,benign,packet_ground_truth,packet\n",
        encoding="utf-8",
    )

    resolved = resolve_eth_packet_csv(str(data_dir), "eth_demo_original_images.npy")
    assert resolved is not None
    assert resolved.endswith("eth_demo_original.csv")


def test_plan_pairs_stays_within_split_and_warns_on_missing_normal_can():
    pair_specs, warnings = _plan_pairs(
        eth_files=["eth_attack_images.npy", "eth_normal_images.npy"],
        can_files=["can_dos_train.csv"],
        pairing_mode="label_cartesian",
    )

    assert pair_specs == [("eth_attack_images.npy", "can_dos_train.csv", 1)]
    assert any("missing within-split normal CAN coverage" in warning for warning in warnings)


def test_standardize_can_dataframe_rescales_byte_normalized_dlc():
    df = pd.DataFrame(
        {
            "CAN_ID": [0.1, 0.2],
            "DLC": [8.0 / 255.0, 8.0 / 255.0],
            "Label": [0, 1],
            **{f"D{i}": [0.0, 1.0] for i in range(8)},
        }
    )

    standardized = standardize_can_dataframe(df)
    assert standardized["DLC"].tolist() == [1.0, 1.0]


def test_build_eth_image_windows_from_packet_csv(tmp_path):
    packet_csv = tmp_path / "eth_demo_replica_packets.csv"
    packet_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n"
        "1,1,1000,128,128\n"
        "2,1,2000,256,256\n",
        encoding="utf-8",
    )

    windows = build_eth_image_windows(str(packet_csv), eth_window_size=1, eth_overlap=0)
    assert windows.shape == (3, 64, 64)
    assert float(windows[0].sum()) > 0.0
    assert not np.allclose(windows[0], windows[2])


def test_standardize_eth_packet_dataframe_handles_missing_optional_columns():
    df = pd.DataFrame(
        {
            "timestamp_sec": [1, 1, 1],
            "captured_len": [64, 128, 256],
            "Label": [0, 1, 1],
            "session_id": ["s1", "s1", "s1"],
            "attack_type": ["benign", "avtp_injection", "avtp_injection"],
            "label_source": ["packet_ground_truth", "packet_ground_truth", "packet_ground_truth"],
            "label_granularity": ["packet", "packet", "packet"],
        }
    )

    standardized = standardize_eth_packet_dataframe(df)
    assert standardized["captured_len"].tolist() == [64.0, 128.0, 256.0]
    assert standardized["original_len"].tolist() == [64.0, 128.0, 256.0]
    assert standardized["Label"].tolist() == [0, 1, 1]
    assert standardized["session_id"].tolist() == ["s1", "s1", "s1"]


def test_correlated_dataset_supports_metadata_eth_representation_without_npy(tmp_path):
    can_csv = tmp_path / "can.csv"
    can_csv.write_text(
        "Timestamp,CAN_ID,DLC,D0,D1,D2,D3,D4,D5,D6,D7,Label\n"
        "0.000,0.1,1.0,0,0,0,0,0,0,0,0,0\n"
        "0.001,0.2,1.0,0,0,0,0,0,0,0,0,0\n"
        "0.002,0.1,1.0,0,0,0,0,0,0,0,0,1\n"
        "0.003,0.2,1.0,0,0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    eth_packet_csv = tmp_path / "eth_demo_replica_packets.csv"
    eth_packet_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,1,0,64,64\n"
        "1,1,1000,128,128\n"
        "2,1,2000,256,256\n"
        "3,1,3000,512,512\n",
        encoding="utf-8",
    )
    eth_label_csv = tmp_path / "eth_demo.csv"
    eth_label_csv.write_text(
        "Label,session_id,attack_type,label_source,label_granularity\n"
        "0,sess_demo,benign,packet_ground_truth,packet\n"
        "0,sess_demo,benign,packet_ground_truth,packet\n"
        "1,sess_demo,avtp_injection,packet_ground_truth,packet\n"
        "1,sess_demo,avtp_injection,packet_ground_truth,packet\n",
        encoding="utf-8",
    )

    ds = CorrelatedHybridVehicleDataset(
        can_csv_path=str(can_csv),
        eth_packet_csv_path=str(eth_packet_csv),
        eth_npy_path=None,
        can_features=STANDARD_CAN_FEATURES_16,
        can_window_size=2,
        can_overlap=1,
        eth_window_size=1,
        eth_overlap=0,
        tolerance_ms=1000.0,
        eth_label_csv_path=str(eth_label_csv),
    )

    assert len(ds) > 0
    sample = ds[0]
    assert tuple(sample["can"].shape) == (2, 16)
    assert tuple(sample["eth"].shape) == (1, 1, 64, 64)


def test_correlate_can_eth_uses_provided_can_dataframe_timestamps(tmp_path):
    can_csv = tmp_path / "can_demo.csv"
    can_csv.write_text(
        "CAN_ID,DLC,D0,D1,D2,D3,D4,D5,D6,D7,Label\n"
        "0.1,1,0,0,0,0,0,0,0,0,0\n"
        "0.2,1,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    can_df = pd.DataFrame(
        {
            "Timestamp": [10.000, 10.100],
            "CAN_ID": [0.1, 0.2],
            "DLC": [1.0, 1.0],
            **{f"D{i}": [0.0, 0.0] for i in range(8)},
            "Label": [0, 0],
        }
    )
    eth_csv = tmp_path / "eth_demo_replica_packets.csv"
    eth_csv.write_text(
        "packet_index,timestamp_sec,timestamp_usec,captured_len,original_len\n"
        "0,10,100000,64,64\n",
        encoding="utf-8",
    )

    pairs_df, report = correlate_can_eth(
        can_csv_path=str(can_csv),
        eth_csv_path=str(eth_csv),
        can_window_size=2,
        can_overlap=0,
        eth_window_size=1,
        eth_overlap=0,
        tolerance_ms=5.0,
        time_mode="absolute",
        can_df=can_df,
    )

    assert len(pairs_df) == 1
    assert report.matched == 1
