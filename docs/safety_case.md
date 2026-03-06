# ISO-Aligned Safety Case (Research Scope)

This repository is a research implementation and is not an ISO 26262 certified product.
The following artifacts provide ISO-aligned safety evidence expected before certification programs.

## Hazard Analysis (Abbreviated)

| Hazard | Cause | Effect | Mitigation |
|---|---|---|---|
| Missed intrusion alert (false negative) | Model uncertainty or concept drift | Unsafe command may pass undetected | Conservative thresholds, heavy-route fallback, periodic recalibration |
| Spurious alert flood (false positive) | Domain shift or bad calibration | Driver/system nuisance and trust degradation | FPR budget in config, threshold calibration, benchmark gating |
| Runtime deadline miss | CPU load spikes | Delayed mitigation decision | Watchdog monitor and fail-safe modes |
| Corrupt model artifact | Tampering or deployment mismatch | Unpredictable detection behavior | SHA256 integrity verification before load |
| Inference backend failure | ONNX runtime failure | Loss of IDS detection | Automatic PyTorch fallback and degraded mode |

## Fallback Policy

- `fail_open`: force `NORMAL` prediction for availability-prioritized scenarios.
- `fail_closed`: force `MALICIOUS` prediction for safety-prioritized scenarios.
- `degraded`: continue with constrained operation while emitting health reason.

## Required Verification Before Vehicle Trials

1. Run `validate_ids` with signed model hash.
2. Run `benchmark_ids` under expected and peak traffic conditions.
3. Run coordinated replay and robustness evaluation.
4. Verify alert egress path (CAN or SOME/IP) on target gateway.
5. Review compliance matrix and unresolved `PARTIAL` items.
