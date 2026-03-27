"""
tests/test_eth_payload_extraction.py

Phase 1 DPI – unit tests covering the full Ethernet payload extraction
and image-encoding pipeline.

Tests verify:
 1. standardize_eth_packet_dataframe() reads, normalises, and zero-fills
    payload columns (backwards-compatible with old metadata-only CSVs).
 2. _eth_window_summary() returns the correct feature vector length and
    provides non-zero payload features when payload columns are present.
 3. encode_eth_window_to_image() produces a (64, 64) float32 image.
 4. build_eth_image_windows() works end-to-end with a synthetic CSV that
    includes DPI payload columns.
"""

import numpy as np
import pandas as pd
import pytest

from src_replica.preprocessing_standard import (
    STANDARD_ETH_IMAGE_SIZE,
    STANDARD_ETH_PAYLOAD_BYTES,
    STANDARD_ETH_PAYLOAD_COLS,
    build_eth_image_windows,
    encode_eth_window_to_image,
    standardize_eth_packet_dataframe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata_only_df(n: int = 4) -> pd.DataFrame:
    """Minimal packet CSV without DPI columns (legacy / old format)."""
    sizes = [64, 128, 256, 512]
    labels = [0, 0, 1, 1]
    attacks = ["benign", "benign", "avtp_injection", "avtp_injection"]
    return pd.DataFrame({
        "timestamp_sec": list(range(n)),
        "timestamp_usec": [0] * n,
        "captured_len": [sizes[i % len(sizes)] for i in range(n)],
        "original_len": [sizes[i % len(sizes)] for i in range(n)],
        "Label": [labels[i % len(labels)] for i in range(n)],
        "session_id": ["s1"] * n,
        "attack_type": [attacks[i % len(attacks)] for i in range(n)],
        "label_source": ["packet_ground_truth"] * n,
        "label_granularity": ["packet"] * n,
    })


def _make_dpi_df(n: int = 4) -> pd.DataFrame:
    """Packet CSV that includes DPI payload columns (new Phase 1 format)."""
    df = _make_metadata_only_df(n)
    df["eth_type"] = 0x0800 / 65535.0
    df["ip_proto"] = 17.0 / 255.0
    df["src_port"] = 2048.0 / 65535.0
    df["dst_port"] = 17220.0 / 65535.0
    df["payload_len"] = [40 + (i % 4) * 20 for i in range(n)]
    entropies = [0.1, 0.2, 0.85, 0.9]
    df["payload_entropy"] = [entropies[i % len(entropies)] for i in range(n)]
    for i, col in enumerate(STANDARD_ETH_PAYLOAD_COLS):
        df[col] = float(i * 10) / 255.0
    return df


# ---------------------------------------------------------------------------
# Test 1: backward compatibility – metadata-only CSV has zero-filled payload
# ---------------------------------------------------------------------------

class TestStandardizeEthPacketDataframeBackwardCompat:
    def test_payload_columns_zero_filled_when_absent(self):
        df = _make_metadata_only_df()
        result = standardize_eth_packet_dataframe(df)
        for col in STANDARD_ETH_PAYLOAD_COLS:
            assert col in result.columns, f"Column {col} missing from standardized output"
            assert float(result[col].sum()) == 0.0, f"Expected zeros for {col} in metadata-only CSV"

    def test_dpi_scalar_columns_zero_filled_when_absent(self):
        df = _make_metadata_only_df()
        result = standardize_eth_packet_dataframe(df)
        for col in ["eth_type", "ip_proto", "src_port", "dst_port", "payload_entropy", "payload_len_norm"]:
            assert col in result.columns
            assert float(result[col].sum()) == 0.0

    def test_label_and_provenance_preserved(self):
        df = _make_metadata_only_df()
        result = standardize_eth_packet_dataframe(df)
        assert result["Label"].tolist() == [0, 0, 1, 1]
        assert "session_id" in result.columns
        assert result["session_id"].tolist() == ["s1", "s1", "s1", "s1"]


# ---------------------------------------------------------------------------
# Test 2: DPI CSV correctly normalised
# ---------------------------------------------------------------------------

class TestStandardizeEthPacketDataframeDPI:
    def test_payload_byte_values_clipped_to_01(self):
        df = _make_dpi_df()
        result = standardize_eth_packet_dataframe(df)
        for col in STANDARD_ETH_PAYLOAD_COLS:
            vals = result[col].to_numpy(dtype=float)
            assert vals.min() >= 0.0, f"{col} has values < 0"
            assert vals.max() <= 1.0, f"{col} has values > 1"

    def test_payload_entropy_non_zero_for_injected_packets(self):
        df = _make_dpi_df()
        result = standardize_eth_packet_dataframe(df)
        assert result["payload_entropy"].iloc[2] > 0.5  # injected rows have high entropy

    def test_ip_proto_normalised(self):
        df = _make_dpi_df(n=2)
        result = standardize_eth_packet_dataframe(df)
        expected = 17.0 / 255.0
        assert pytest.approx(result["ip_proto"].iloc[0], abs=1e-5) == expected

    def test_payload_len_norm_clipped(self):
        df = _make_dpi_df()
        result = standardize_eth_packet_dataframe(df)
        assert result["payload_len_norm"].max() <= 1.0


# ---------------------------------------------------------------------------
# Test 3: _eth_window_summary feature vector dimensions
# ---------------------------------------------------------------------------

class TestEthWindowSummaryDimensions:
    def test_summary_with_dpi_has_36_features(self):
        """36 = 16 metadata + 16 payload byte means + 4 DPI scalar stats."""
        from src_replica.preprocessing_standard import _eth_window_summary
        df = _make_dpi_df()
        std_df = standardize_eth_packet_dataframe(df)
        features = _eth_window_summary(std_df)
        # 16 metadata + 20 DPI (16 byte means + entropy mean/std/max + ip_proto)
        assert len(features) == 36, f"Expected 36 features, got {len(features)}"

    def test_summary_without_dpi_returns_zeros_for_payload_features(self):
        from src_replica.preprocessing_standard import _eth_window_summary
        df = _make_metadata_only_df()
        std_df = standardize_eth_packet_dataframe(df)
        features = _eth_window_summary(std_df)
        assert len(features) == 36
        # Payload byte means (indices 16..31) should all be 0 for metadata-only data
        assert features[16:32].sum() == 0.0

    def test_summary_payload_features_nonzero_when_dpi_present(self):
        from src_replica.preprocessing_standard import _eth_window_summary
        df = _make_dpi_df()
        std_df = standardize_eth_packet_dataframe(df)
        features = _eth_window_summary(std_df)
        assert features[16:32].sum() > 0.0, "Payload byte means should be nonzero for DPI data"


# ---------------------------------------------------------------------------
# Test 4: encode_eth_window_to_image produces 64×64
# ---------------------------------------------------------------------------

class TestEncodeEthWindowToImage:
    def test_image_shape_is_64x64(self):
        df = _make_dpi_df()
        std_df = standardize_eth_packet_dataframe(df)
        img = encode_eth_window_to_image(std_df)
        assert img.shape == (STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE), (
            f"Expected ({STANDARD_ETH_IMAGE_SIZE}, {STANDARD_ETH_IMAGE_SIZE}), got {img.shape}"
        )

    def test_image_values_in_01(self):
        df = _make_dpi_df()
        std_df = standardize_eth_packet_dataframe(df)
        img = encode_eth_window_to_image(std_df)
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_dpi_image_differs_from_metadata_only_image(self):
        """DPI data must produce a different image than metadata-only data."""
        meta_df = standardize_eth_packet_dataframe(_make_metadata_only_df())
        dpi_df = standardize_eth_packet_dataframe(_make_dpi_df())
        img_meta = encode_eth_window_to_image(meta_df)
        img_dpi = encode_eth_window_to_image(dpi_df)
        assert not np.allclose(img_meta, img_dpi), (
            "DPI image must differ from metadata-only image (payload bytes should change the image)"
        )


# ---------------------------------------------------------------------------
# Test 5: build_eth_image_windows end-to-end with DPI CSV
# ---------------------------------------------------------------------------

class TestBuildEthImageWindowsEndToEnd:
    def test_builds_windows_with_dpi_csv(self, tmp_path):
        csv_path = tmp_path / "eth_demo_replica_packets.csv"
        df = _make_dpi_df(n=10)
        df.to_csv(csv_path, index=False)

        windows = build_eth_image_windows(
            str(csv_path),
            eth_window_size=3,
            eth_overlap=0,
        )
        assert windows.ndim == 3
        assert windows.shape[1] == STANDARD_ETH_IMAGE_SIZE
        assert windows.shape[2] == STANDARD_ETH_IMAGE_SIZE
        assert windows.shape[0] >= 1

    def test_backward_compat_metadata_only_csv(self, tmp_path):
        csv_path = tmp_path / "eth_legacy_replica_packets.csv"
        df = _make_metadata_only_df(n=6)
        df.to_csv(csv_path, index=False)

        windows = build_eth_image_windows(str(csv_path), eth_window_size=2, eth_overlap=0)
        assert windows.shape == (3, STANDARD_ETH_IMAGE_SIZE, STANDARD_ETH_IMAGE_SIZE)
        assert windows.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 6: STANDARD_ETH_IMAGE_SIZE constant is 64
# ---------------------------------------------------------------------------

def test_standard_eth_image_size_is_64():
    assert STANDARD_ETH_IMAGE_SIZE == 64, (
        "STANDARD_ETH_IMAGE_SIZE should be 64 after Phase 1 DPI upgrade"
    )


def test_standard_eth_payload_cols_count():
    assert len(STANDARD_ETH_PAYLOAD_COLS) == STANDARD_ETH_PAYLOAD_BYTES == 16
