from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class AlertRecord:
    timestamp: str
    source: str
    attack_class: str
    confidence: float
    route_path: str
    latency_ms: float
    decision_id: str
    details: Optional[Dict[str, Any]] = None

    @staticmethod
    def now(
        source: str,
        attack_class: str,
        confidence: float,
        route_path: str,
        latency_ms: float,
        decision_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> "AlertRecord":
        return AlertRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            attack_class=attack_class,
            confidence=float(confidence),
            route_path=route_path,
            latency_ms=float(latency_ms),
            decision_id=decision_id,
            details=details or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    fpr: float
    fnr: float
    mcc: float
    cpu_percent: float
    memory_mb: float
    power_watts: Optional[float]
    hardware_id: str
    os: str
    model_hash: str
    total_samples: int
    deadline_ms: float
    deadline_miss_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
