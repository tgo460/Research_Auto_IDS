import json
import os
import socket
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd


class CanIngest(ABC):
    @abstractmethod
    def read_frame(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


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
    def __init__(self, csv_path: str, timestamp_col: str = "Timestamp"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self._df = pd.read_csv(csv_path)
        self._idx = 0
        self._timestamp_col = timestamp_col

    def read_frame(self) -> Optional[Dict[str, Any]]:
        if self._idx >= len(self._df):
            return None
        row = self._df.iloc[self._idx]
        self._idx += 1
        payload = [float(row.get(f"D{i}", 0.0)) for i in range(8)]
        return {
            "timestamp": float(row.get(self._timestamp_col, self._idx)),
            "can_id": int(row.get("CAN_ID", 0)),
            "dlc": int(row.get("DLC", 8)),
            "payload": payload,
            "label": int(row.get("Label", 0)),
        }


class CsvEthIngest(EthIngest):
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self._df = pd.read_csv(csv_path)
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
            "label": 0,
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
            "can_id": int(msg.arbitration_id),
            "dlc": int(msg.dlc),
            "payload": [float(x) / 255.0 for x in data],
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


def to_can_window(frames: Deque[Dict[str, Any]], expected_size: int) -> np.ndarray:
    if len(frames) < expected_size:
        raise ValueError("insufficient CAN frames")
    win = list(frames)[-expected_size:]
    out = []
    for fr in win:
        out.append(
            [
                float(fr.get("can_id", 0.0)),
                float(fr.get("dlc", 0.0)),
                *[float(x) for x in fr.get("payload", [0.0] * 8)],
            ]
        )
    return np.asarray(out, dtype=np.float32)


def to_eth_image(frame: Dict[str, Any], size: int = 32) -> np.ndarray:
    # Deterministic lightweight encoding for runtime fallback when packet images are absent.
    captured = float(frame.get("captured_len", 0.0))
    original = float(frame.get("original_len", 1.0))
    ratio = 0.0 if original <= 0 else min(captured / original, 1.5)
    base = np.full((1, size, size), ratio, dtype=np.float32)
    return base
