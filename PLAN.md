# Strict Gap-Closure Checklist: CAN+Ethernet IDS vs PDF Standards

## Summary
This checklist closes gaps from current research prototype to deployment-ready, standards-aligned implementation for in-vehicle CAN+Ethernet IDS.  
Completion rule: every item below must be marked `PASS` with listed artifact evidence.

## Public API / Interface Changes Required
1. Add runtime adapter interfaces: `CanIngest`, `EthIngest`, `AlertEgress`, `HealthMonitor`.
2. Add deployment config schema (`deployment.yaml` or JSON) with fixed keys: model paths, window sizes, routing threshold, latency budget, FPR budget, fail-safe mode, integrity hash.
3. Add stable runtime CLI commands: `run_ids`, `benchmark_ids`, `validate_ids`, `replay_can_eth`.
4. Add alert schema contract (JSON/CAN payload): `timestamp`, `source`, `attack_class`, `confidence`, `route_path`, `latency_ms`, `decision_id`.
5. Add benchmark report schema with mandatory fields: `latency_p50/p95/max`, `fpr`, `fnr`, `mcc`, `cpu`, `memory`, `power`, `hardware_id`, `os`, `model_hash`.

## Strict Checklist
| ID | Requirement | Current Gap | Closure Action | PASS Evidence |
|---|---|---|---|---|
| C01 | Training pipeline correctness | Light training script uses `datasets_list` before init | Fix ordering bug and add unit test for trainer startup | Test proving trainer runs from clean checkout |
| C02 | ONNX inference contract consistency | ONNX test input shape mismatch vs export | Unify canonical shapes and enforce shape checks at load | `test_onnx` green with real model |
| C03 | Unified window standard | Mixed 10/32/100 CAN window usage in reports/scripts | Set one production window policy and enforce globally | Config + CI check rejecting mismatched window sizes |
| C04 | Real CAN integration | Offline CSV only | Add live CAN input via `python-can/socketcan` (or target stack) | Integration test consuming live/simulated CAN bus frames |
| C05 | Real Ethernet integration | Offline packet/image pipeline only | Add live Ethernet flow ingest and feature extraction | Integration test on pcap/live stream with deterministic windows |
| C06 | In-vehicle egress integration | No standard alert publish path | Publish alerts to CAN and/or SOME/IP endpoint per config | End-to-end demo from ingest to vehicle-side alert consumer |
| C07 | ONNX edge runtime hardening | Python path exists but no deployment-hardened runtime contract | Lock ORT session options, input guards, timeout handling | Runtime validation report with rejected malformed inputs |
| C08 | Quantization requirement | Quant stubs exist but no actual int8/fp16 quantized artifact | Produce calibrated quantized model and benchmark it | Quantized model artifact + latency/accuracy delta report |
| C09 | ECU target validation | Benchmarks on Windows Intel only | Benchmark on target ARM ECU or faithful emulator | Report with hardware/OS fingerprint and latency/resource stats |
| C10 | Real-time deadline proof | Latency measured, but no worst-case under load proof | Run peak-load tests with deadline miss accounting | Worst-case latency and miss-rate report vs `<100ms` target |
| C11 | Detection target compliance | Latest FPR around 9.64% (>5%) | Retrain/calibrate until CAN/Ethernet/combined meet target | Final report showing per-domain FPR `<5%` |
| C12 | Metric completeness | No mandatory MCC/FNR in final KPI | Add MCC/FNR to all evaluation and deployment reports | Updated report schema and generated metrics files |
| C13 | Cross-network attack validation | No explicit Ethernet→CAN coordinated attack test suite | Add coordinated scenario replay tests | Scenario report with detection rates and delay per scenario |
| C14 | Safety fail-safe behavior | No formal fallback/watchdog policy | Implement fail-open/fail-safe policy, watchdog, degraded mode | Fault injection tests proving safe fallback activation |
| C15 | Security/integrity controls | No runtime model integrity verification path | Verify model hash/signature before load; refuse mismatch | Startup log + negative test with tampered model |
| C16 | Adversarial robustness | No evasion robustness benchmark | Add adversarial/evasion test harness and thresholding policy | Robustness report with degradation bounds |
| C17 | Reproducibility contract | Referenced eval command/config path missing in repo | Add canonical `configs/` and top-level reproducible entrypoint | One-command reproducible run generating all key reports |
| C18 | Compliance documentation | No traceability matrix to PDF requirements | Add requirement-to-evidence matrix | `compliance_matrix.md` with links to artifacts/tests |

## Test Cases and Scenarios (Must Exist Before Sign-off)
1. Unit tests for model I/O shape validation, threshold logic, and fallback routing.
2. Integration tests for live CAN ingest, live Ethernet ingest, and alert egress.
3. Performance tests at nominal and peak load with p50/p95/max latency and deadline misses.
4. Detection tests for CAN attacks (DoS, fuzzy, spoof/impersonation).
5. Detection tests for Ethernet attacks (scan, flood, unauthorized access class used in dataset).
6. Coordinated attack tests (Ethernet trigger followed by CAN anomaly).
7. Fault-injection tests (missing model, malformed frames, runtime timeout, heavy-model failure).
8. Security tests (tampered model hash/signature rejection).
9. Robustness tests (adversarial/noisy inputs and concept-drift slice).
10. Cross-vehicle/generalization tests using unseen split policy.

## Assumptions and Defaults
1. Compliance target is the attached PDF checklist, treated as strict acceptance criteria.
2. Default pass thresholds: combined latency `<100 ms` per decision window, per-domain FPR `<5%`, and explicit MCC/FNR reporting.
3. Initial production runtime can be Python + ONNX Runtime, but C++ ORT path is required for full automotive deployment claim.
4. Target deployment platform is ARM-based ECU/gateway; desktop-only validation is not sufficient.
5. If ISO 26262 certification is out of scope, minimum acceptable output is ISO-aligned safety case artifacts (hazard analysis, fail-safe tests, traceability).

## Definition of Done
All `C01`–`C18` are `PASS`, required tests are automated, and one signed compliance package is produced containing reports, configs, model hashes, and traceability matrix.
