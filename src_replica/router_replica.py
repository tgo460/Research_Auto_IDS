from dataclasses import dataclass, asdict
from typing import Dict
import numpy as np
import torch

@dataclass
class RouterConfig:
    threshold: float = 0.6
    mode: str = 'max_softmax'
    route_if_below_or_equal: bool = True

class ConfidenceRouter:
    def __init__(self, config: RouterConfig):
        self.config = config

    @staticmethod
    def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return conf

    def route_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        conf = self.confidence_from_logits(logits)
        if self.config.route_if_below_or_equal:
            return conf <= self.config.threshold
        else:
            return conf < self.config.threshold

    def route_from_confidence(self, conf: torch.Tensor) -> torch.Tensor:
        if self.config.route_if_below_or_equal:
            return conf <= self.config.threshold
        else:
            return conf < self.config.threshold

    def describe(self) -> Dict[str, object]:
        return asdict(self.config)

def tune_threshold_by_quantile(confidences: np.ndarray, route_fraction: float) -> float:
    if confidences.size == 0:
        return 0.5
    route_fraction = float(np.clip(route_fraction, 0.0, 1.0))
    q = route_fraction
    return float(np.quantile(confidences, q))
