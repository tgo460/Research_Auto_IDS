# Compliance Matrix: Automotive IDS Gap Closure

This matrix maps checklist items `C01`-`C18` to implementation evidence in this repository.

| ID | Status | Evidence |
|---|---|---|
| C01 | PASS | `src_replica/train_improved_light_model.py`, `tests/test_c01_training_startup.py` |
| C02 | PASS | `src_replica/test_onnx_replica.py`, `src_replica/realtime_inference_replica.py`, `tests/test_c02_onnx_contract.py` |
| C03 | PASS | `src_replica/runtime/standards.py`, `src_replica/runtime/config.py`, `configs/deployment.schema.json`, `tests/test_runtime_compliance.py::test_window_standard_enforced` |
| C04 | PARTIAL | `src_replica/runtime/adapters.py::SocketCanIngest`, `tests/test_live_adapters.py` (requires real CAN interface for full validation) |
| C05 | PARTIAL | `src_replica/runtime/adapters.py::CsvEthIngest/PcapEthIngest`, `src_replica/runtime/engine.py`, `tests/test_runtime_integration.py` |
| C06 | PASS | `src_replica/runtime/contracts.py::AlertRecord`, `src_replica/runtime/adapters.py::FileAlertEgress/StdoutAlertEgress`, `src_replica/ids_cli.py` |
| C07 | PASS | `src_replica/realtime_inference_replica.py::_validate_onnx_contract`, strict shape guards in `process_packet` |
| C08 | PARTIAL | `src_replica/quantize_onnx_replica.py` implemented; current environment lacks working `onnx` quantization dependency (`onnx.defs` missing) |
| C09 | PARTIAL | Benchmark schema supports ECU fingerprinting (`hardware_id`, `os`) in `src_replica/runtime/contracts.py`; actual ARM ECU run required externally |
| C10 | PASS | Deadline miss tracking in `src_replica/runtime/metrics.py` and watchdog policy in `src_replica/runtime/adapters.py` |
| C11 | PARTIAL | Framework supports target checks (`fpr_budget` in config/report); meeting `<5%` is model/data dependent and must be achieved by retraining |
| C12 | PASS | Added `fnr`/`mcc` to `evaluate.py` and benchmark report schema in `src_replica/runtime/contracts.py` |
| C13 | PARTIAL | `src_replica/coordinated_attack_replay.py` + `tests/test_coordinated_attack.py` (synthetic coordinated trigger; full HIL scenario still required) |
| C14 | PASS | Fail-safe modes (`fail_open`, `fail_closed`, `degraded`) in `src_replica/runtime/engine.py` + watchdog in adapters |
| C15 | PASS | SHA256 integrity verification in `src_replica/runtime/security.py`, tests in `tests/test_runtime_compliance.py` |
| C16 | PARTIAL | Runtime hooks exist for robustness testing; dedicated adversarial benchmark script still needed for complete coverage |
| C17 | PASS | Added `configs/repro_full_and_v2.json` and top-level `evaluate.py` reproducible command |
| C18 | PASS | This file (`compliance_matrix.md`) + test and CLI artifact references |

## Notes
- Items marked `PARTIAL` require external execution context (ARM ECU hardware and retraining outcomes) or extended benchmark suites.
