import json
import os
import socket
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd

from src_replica.preprocessing_standard import (
    BYTE_COLS,
    STANDARD_CAN_FEATURES_16,
    encode_eth_frame_dict_to_image,
    normalize_can_id_series,
    normalize_dlc_series,
    normalize_payload_series,
    validate_eth_label_dataframe,
)


class CanIngest(ABC):
    @abstractmethod
    def read_frame(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
        
class VirtualCanIngest(CanIngest):
    """
    Virtual CAN simulator using vcan.
    Ideal for testing when hardware CAN components are unavailable.
    It reads raw CAN messages from a virtual bus natively supported by Linux/Raspberry Pi.
    """
    def __init__(self, interface: str = 'vcan0'):
        import can
        self.interface = interface
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"[VirtualCanIngest] Connected to virtual bus: {interface}")
        except Exception as e:
            print(f"[VirtualCanIngest] Failed to bind to {interface}. Ensure vcan is configured.")
            self.bus = None

    def read_frame(self) -> Optional[Dict[str, Any]]:
        if not self.bus:
            return None
        
        msg = self.bus.recv(timeout=1.0)
        if msg:
            return {
                "timestamp": msg.timestamp,
                "can_id": msg.arbitration_id,
                "dlc": msg.dlc,
                "payload": [float(x) / 255.0 for x in msg.data[:8]],
            }
        return None


class EthIngest(ABC):
    @abstractmethod
    def read_frame(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class AlertEgress(ABC):
    @abstractmethod
    def publish(self, alert: Dict[str, Any]) -> None:
        raise NotImplementedError


class HealthMonitor(ABC):
    @abstractmethod
    def heartbeat(self, latency_ms: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def tripped(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def last_reason(self) -> str:
        raise NotImplementedError


class CsvCanIngest(CanIngest):
    def __init__(self, csv_path: str, timestamp_col: str = "Timestamp", start_row: int = 0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self._df = pd.read_csv(csv_path)
        if start_row > 0:
            self._df = self._df.iloc[int(start_row):].reset_index(drop=True)
        self._idx = 0
        self._timestamp_col = timestamp_col

    _ENGINEERED_COLS = STANDARD_CAN_FEATURES_16[10:]

    def read_frame(self) -> Optional[Dict[str, Any]]:
        if self._idx >= len(self._df):
            return None
        row = self._df.iloc[self._idx]
        self._idx += 1
        payload = [
            float(normalize_payload_series(pd.Series([row.get(f"D{i}", 0.0)])).iloc[0])
            for i in range(8)
        ]
        frame: Dict[str, Any] = {
            "timestamp": float(row.get(self._timestamp_col, self._idx)),
            "can_id": float(normalize_can_id_series(pd.Series([row.get("CAN_ID", 0.0)])).iloc[0]),
            "dlc": float(normalize_dlc_series(pd.Series([row.get("DLC", 8.0)])).iloc[0]),
            "payload": payload,
            "label": int(row.get("Label", 0)),
        }
        # Carry engineered features if present in the CSV
        eng = [float(row.get(c, 0.0)) for c in self._ENGINEERED_COLS if c in row.index]
        if eng:
            frame["engineered"] = eng
        return frame


class CsvEthIngest(EthIngest):
    def __init__(self, csv_path: str, start_row: int = 0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self._df = pd.read_csv(csv_path)
        if start_row > 0:
            self._df = self._df.iloc[int(start_row):].reset_index(drop=True)
        validate_eth_label_dataframe(
            self._df,
            context=f"ETH replay CSV {csv_path}",
            require_label=True,
            require_provenance=True,
        )
        self._idx = 0

    def read_frame(self) -> Optional[Dict[str, Any]]:
        if self._idx >= len(self._df):
            return None
        row = self._df.iloc[self._idx]
        self._idx += 1
        ts_sec = float(row.get("timestamp_sec", self._idx))
        ts_usec = float(row.get("timestamp_usec", 0.0))
        return {
            "timestamp": ts_sec + ts_usec / 1_000_000.0,
            "captured_len": float(row.get("captured_len", 0.0)),
            "original_len": float(row.get("original_len", 0.0)),
            "label": int(row.get("Label", 0)),
            "session_id": str(row.get("session_id", "")),
            "attack_type": str(row.get("attack_type", "")),
            "label_source": str(row.get("label_source", "")),
            "label_granularity": str(row.get("label_granularity", "")),
        }


class PcapEthIngest(EthIngest):
    def __init__(self, pcap_path: str):
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(pcap_path)
        try:
            from scapy.all import rdpcap  # type: ignore
        except Exception as exc:
            raise RuntimeError("scapy is required for PcapEthIngest") from exc
        self._pkts = rdpcap(pcap_path)
        self._idx = 0

    def read_frame(self) -> Optional[Dict[str, Any]]:
        if self._idx >= len(self._pkts):
            return None
        pkt = self._pkts[self._idx]
        self._idx += 1
        plen = float(len(pkt))
        return {
            "timestamp": float(getattr(pkt, "time", self._idx)),
            "captured_len": plen,
            "original_len": plen,
        }


class SocketCanIngest(CanIngest):
    def __init__(self, channel: str = "can0", bustype: str = "socketcan"):
        try:
            import can  # type: ignore
        except Exception as exc:
            raise RuntimeError("python-can is required for SocketCanIngest") from exc
        self._can = can
        self._bus = can.interface.Bus(channel=channel, bustype=bustype)

    def read_frame(self) -> Optional[Dict[str, Any]]:
        msg = self._bus.recv(timeout=0.01)
        if msg is None:
            return None
        data = list(msg.data)[:8]
        while len(data) < 8:
            data.append(0)
        return {
            "timestamp": float(msg.timestamp),
            "can_id": float(normalize_can_id_series(pd.Series([msg.arbitration_id])).iloc[0]),
            "dlc": float(normalize_dlc_series(pd.Series([msg.dlc])).iloc[0]),
            "payload": [float(normalize_payload_series(pd.Series([x])).iloc[0]) for x in data],
            "label": 0,
        }


class FileAlertEgress(AlertEgress):
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def publish(self, alert: Dict[str, Any]) -> None:
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")


class StdoutAlertEgress(AlertEgress):
    def publish(self, alert: Dict[str, Any]) -> None:
        print(json.dumps(alert, ensure_ascii=True))


class CanAlertEgress(AlertEgress):
    def __init__(self, channel: str = "can0", bustype: str = "socketcan", arbitration_id: int = 0x6A0):
        try:
            import can  # type: ignore
        except Exception as exc:
            raise RuntimeError("python-can is required for CanAlertEgress") from exc
        self._can = can
        self._bus = can.interface.Bus(channel=channel, bustype=bustype)
        self._arb_id = int(arbitration_id)

    def publish(self, alert: Dict[str, Any]) -> None:
        # Compact payload: class/confidence/latency in fixed 8-byte message.
        cls = 1 if str(alert.get("attack_class", "NORMAL")).upper() == "MALICIOUS" else 0
        conf = int(max(0, min(255, round(float(alert.get("confidence", 0.0)) * 255))))
        lat = int(max(0, min(65535, round(float(alert.get("latency_ms", 0.0))))))
        payload = [cls, conf, lat & 0xFF, (lat >> 8) & 0xFF, 0, 0, 0, 0]
        msg = self._can.Message(arbitration_id=self._arb_id, data=payload, is_extended_id=False)
        self._bus.send(msg)


class SomeIPEgress(AlertEgress):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, alert: Dict[str, Any]) -> None:
        # SOME/IP production integration should replace this with protocol-compliant serializer.
        data = json.dumps(alert, ensure_ascii=True).encode("utf-8")
        self.sock.sendto(data, (self.host, self.port))


class WatchdogHealthMonitor(HealthMonitor):
    def __init__(
        self,
        latency_budget_ms: float,
        max_consecutive_misses: int = 3,
        min_heartbeat_hz: float = 0.5,
    ):
        self.latency_budget_ms = float(latency_budget_ms)
        self.max_consecutive_misses = int(max_consecutive_misses)
        self.min_heartbeat_hz = float(min_heartbeat_hz)
        self._misses = 0
        self._last_reason = ""
        self._history: Deque[float] = deque(maxlen=32)
        self._last_t = time.time()

    def heartbeat(self, latency_ms: float) -> None:
        now = time.time()
        self._history.append(float(latency_ms))
        if latency_ms > self.latency_budget_ms:
            self._misses += 1
            self._last_reason = (
                f"latency budget exceeded: {latency_ms:.3f}>{self.latency_budget_ms:.3f}"
            )
        else:
            self._misses = 0

        interval = max(now - self._last_t, 1e-6)
        self._last_t = now
        if (1.0 / interval) < self.min_heartbeat_hz:
            self._misses += 1
            self._last_reason = "heartbeat rate below minimum"

    def tripped(self) -> bool:
        return self._misses >= self.max_consecutive_misses

    def last_reason(self) -> str:
        return self._last_reason


def _payload_entropy(payload: np.ndarray) -> float:
    payload_i = np.clip(payload * 255.0, 0, 255).astype(np.int32)
    hist, _ = np.histogram(payload_i, bins=16, range=(0, 256), density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(min(len(payload_i), 16))) if len(payload_i) > 1 else 1.0
    return 0.0 if max_entropy <= 0 else float(entropy / max_entropy)


def _compute_engineered_can_features(
    frames: Deque[Dict[str, Any]],
    rolling_window: int = 200,
) -> np.ndarray:
    win = list(frames)
    n = len(win)
    can_ids = [round(float(fr.get("can_id", 0.0)), 6) for fr in win]
    timestamps = [float(fr.get("timestamp", idx)) for idx, fr in enumerate(win)]
    payloads = [
        np.asarray(fr.get("payload", [0.0] * 8), dtype=np.float32)[:8]
        for fr in win
    ]

    inter_arrival_raw = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        inter_arrival_raw[i] = max(0.0, timestamps[i] - timestamps[i - 1])

    inter_arrival = np.zeros(n, dtype=np.float32)
    max_so_far = 0.0
    for i, delta in enumerate(inter_arrival_raw):
        max_so_far = max(max_so_far, float(delta))
        scale = max(max_so_far, 1e-9)
        inter_arrival[i] = float(delta / scale)

    switches = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        switches[i] = 1.0 if abs(can_ids[i] - can_ids[i - 1]) > 1e-6 else 0.0

    counts: Dict[float, int] = {}
    out = np.zeros((n, 6), dtype=np.float32)

    for i, can_id in enumerate(can_ids):
        counts[can_id] = counts.get(can_id, 0) + 1
        start = max(0, i - rolling_window + 1)
        prefix_ids = can_ids[start : i + 1]
        prefix_len = max(len(prefix_ids), 1)
        freq_global = counts[can_id] / float(i + 1)
        freq_win = prefix_ids.count(can_id) / float(prefix_len)
        entropy = _payload_entropy(payloads[i])
        ia_roll = float(np.mean(inter_arrival[start : i + 1]))
        switch_rate = float(np.mean(switches[start : i + 1]))
        out[i] = np.asarray(
            [freq_global, freq_win, entropy, inter_arrival[i], ia_roll, switch_rate],
            dtype=np.float32,
        )

    return out


def to_can_window(frames: Deque[Dict[str, Any]], expected_size: int) -> np.ndarray:
    if len(frames) < expected_size:
        raise ValueError("insufficient CAN frames")
    win = list(frames)[-expected_size:]
    has_engineered = all("engineered" in fr for fr in win)
    engineered = None if has_engineered else _compute_engineered_can_features(deque(win))
    out = []
    for idx, fr in enumerate(win):
        row = [
            float(fr.get("can_id", 0.0)),
            float(fr.get("dlc", 0.0)),
            *[float(x) for x in fr.get("payload", [0.0] * 8)],
        ]
        # Use replay-provided engineered features when available; otherwise compute
        # causal window-local features so live inputs do not degrade to zero padding.
        if has_engineered:
            row.extend(fr["engineered"])
        else:
            row.extend(engineered[idx].tolist())
        out.append(row)
    return np.asarray(out, dtype=np.float32)


def to_eth_image(frame: Dict[str, Any], size: int = 32) -> np.ndarray:
    return np.expand_dims(encode_eth_frame_dict_to_image(frame, image_size=size), axis=0)
