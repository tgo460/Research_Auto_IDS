# Research Validity Audit

Date: 2026-03-20

## Scope

This audit evaluates whether the current repository supports unbiased, research-grade claims for a hybrid CAN plus Ethernet IDS. It is based on the local codebase and saved artifacts only. It does not rely on external literature review or internet sources.

This audit is complementary to [PLAN.md](../PLAN.md): the plan tracks engineering and deployment closure, while this document focuses on threats to empirical validity.

## Executive Verdict

The repository contains useful research scaffolding, but it does not yet meet the standard required for strong scientific claims about hybrid CAN plus Ethernet intrusion detection. The strongest current evidence supports low-latency inference, not trustworthy detection quality or robust generalization.

In its current state, the codebase should be described as a research prototype with unresolved validity risks rather than a deployment-ready or publication-ready IDS.

## What Is Already Strong

- Metric coverage is better than many prototypes: FPR, FNR, MCC, calibration, bootstrap intervals, and permutation tests are present in the evaluation code.
- The repository includes split manifests, runtime contracts, malformed-input validation, and model hash checking.
- The repo contains dedicated artifacts for robustness, coordinated replay, generalization, and LOAO-style evaluation, which is a good sign of research intent even when the current implementations are not yet trustworthy.

## Critical Findings

### 1. ETH-only runtime evaluation is invalid

The runtime path expects Ethernet labels to come from a `Label` column in replay CSVs, but the extracted Ethernet smoke CSVs only contain timestamp and packet-length fields.

Evidence:

- [src_replica/runtime/adapters.py](../src_replica/runtime/adapters.py): `CsvEthIngest.read_frame()` defaults missing `Label` to `0`.
- [datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv](../datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv): no `Label` column in the file header.
- [src_replica/coordinated_attack_replay.py](../src_replica/coordinated_attack_replay.py): ETH-only scenarios are explicitly expected to behave as attacks.
- [reports/coordinated_attack_report.json](../reports/coordinated_attack_report.json): ETH-only scenarios show `detection_rate: null`, indicating no attack-labeled samples reached the runtime metric path.

Why this matters:

- The current runtime benchmark cannot substantiate hybrid detection claims for ETH-only or coordinated attack cases.
- Any result derived from replaying injected Ethernet CSVs through the current runtime path is effectively dominated by CAN labels.

### 2. Feature and label leakage are present

The current feature pipeline includes file-global CAN statistics and filename-derived ETH labels.

Evidence:

- [src_replica/features_can_replica.py](../src_replica/features_can_replica.py): `can_id_freq_global` is computed over the full file before any split-aware processing.
- [src_replica/dataloader_correlated_replica.py](../src_replica/dataloader_correlated_replica.py): `eth_label` is derived from whether the filename contains `injected` or `attack`.
- [src_replica/dataloader_correlated_replica.py](../src_replica/dataloader_correlated_replica.py): final labels are merged with `label_policy='max'`, which can turn scenario metadata into supervision.
- [reports/edge_benchmark_combined.json](../reports/edge_benchmark_combined.json): the report explicitly notes file-global engineered features leaking attack context into normal-labeled rows.

Why this matters:

- The model may learn scenario identity rather than intrusion semantics.
- This directly undermines claims about causal detection capability and cross-domain generalization.

### 3. Split integrity is not enforced strongly enough

The repository has split manifests, but the enforcement is shallow and the main cascade evaluation code uses heuristic fallback behavior that can contaminate evaluation.

Evidence:

- [evaluate.py](../evaluate.py): the so-called strict split check only verifies file existence.
- [src_replica/cascade_eval_replica.py](../src_replica/cascade_eval_replica.py): always loads `split_v1`.
- [src_replica/cascade_eval_replica.py](../src_replica/cascade_eval_replica.py): borrows benign or attack CAN files from the training split when the requested split lacks them.
- [src_replica/cascade_eval_replica.py](../src_replica/cascade_eval_replica.py): pairs attack ETH files with the first available attack CAN file instead of using a principled split-locked pairing protocol.

Why this matters:

- Validation and test results are not guaranteed to be independent.
- Reported results may reflect heuristic pairing choices rather than genuine cross-modal performance.

### 4. Training and deployment representations do not match

The trained model expects engineered CAN features and structured ETH inputs, but the live runtime can degrade both modalities substantially.

Evidence:

- [src_replica/runtime/engine.py](../src_replica/runtime/engine.py): pads missing CAN features with zeros when the runtime stream provides fewer than the model expects.
- [src_replica/runtime/adapters.py](../src_replica/runtime/adapters.py): `SocketCanIngest` only emits raw CAN fields, not engineered ones.
- [src_replica/runtime/adapters.py](../src_replica/runtime/adapters.py): `CsvCanIngest` casts `CAN_ID` and `DLC` to integers.
- [datasets/can_dos_train.csv](../datasets/can_dos_train.csv): the training CSV uses normalized float values rather than integer IDs.
- [src_replica/runtime/adapters.py](../src_replica/runtime/adapters.py): `PcapEthIngest` labels all frames as `0`.
- [src_replica/runtime/adapters.py](../src_replica/runtime/adapters.py): `to_eth_image()` collapses Ethernet input to a constant packet-length ratio image, which is much weaker than the training `.npy` representation.

Additional local check:

- In the first 1,000 rows of `datasets/can_dos_train.csv`, 99.5% of `CAN_ID` values are less than `1`, so integer casting collapses nearly all of them to `0` during CSV replay.

Why this matters:

- The repo can report latency on inputs that do not match the training distribution.
- Strong deployment claims are not justified until the runtime feature pipeline matches the training pipeline.

### 5. Saved empirical results contradict the headline narrative

The README still contains high-confidence performance language that is not supported by the latest saved metrics.

Evidence:

- [README.md](../README.md): describes the system as research-grade and deployment-ready.
- [README.md](../README.md): states typical routing of `3-5%`.
- [README.md](../README.md): repeats `100%` detection accuracy claims.
- [reports/final_metrics_latest.json](../reports/final_metrics_latest.json): latest aggregate metrics show poor F1, high FNR, and no cascade improvement.
- [reports/generalization_eval_report.json](../reports/generalization_eval_report.json): cross-domain and attack-holdout metrics collapse to `FPR = 1.0`, `MCC = 0.0`, and tiny sample sizes.
- [reports/loao_evaluation_report.json](../reports/loao_evaluation_report.json): LOAO-style results are based on single-digit attack support in some cases and are not strong evidence of zero-day robustness.

Why this matters:

- The public-facing repository story is currently stronger than the empirical evidence.
- This is a research communication risk as well as a modeling risk.

### 6. The current generalization protocols are too weak for confident claims

The generalization and LOAO scripts are valuable, but the current protocols are not yet strong enough to support robust external-validity claims.

Evidence:

- [src_replica/generalization_eval_replica.py](../src_replica/generalization_eval_replica.py): the report itself notes that the attack-holdout protocol is not true LOAO retraining.
- [src_replica/generalization_eval_replica.py](../src_replica/generalization_eval_replica.py): uses fixed, manually chosen attack/normal pairings with very small resulting datasets.
- [src_replica/loao_train_replica.py](../src_replica/loao_train_replica.py): LOAO retraining exists, but saved support sizes remain very small in the resulting report.

Why this matters:

- The observed failures may be real, but the current protocol does not give enough statistical power to characterize them confidently.
- Future claims need larger support, stronger disjointness, and cleaner attack-family holdout definitions.

## Counter-Hypotheses

These are the main alternatives that should be tested before trusting current results:

- The model is detecting file identity or scenario metadata instead of attack behavior.
- The ETH branch contributes little to runtime detection because ETH supervision disappears in the replay path.
- Apparent CAN plus ETH fusion gains are alignment or dataset artifacts, not genuine cross-protocol reasoning.
- The heavy fallback is not improving outcomes because the routed subset is poorly calibrated or distribution-shifted.
- The live deployment path would underperform sharply relative to offline evaluation because the runtime features do not match the training representation.

## Minimum Bar Before Strong Research Claims

The following should be treated as the minimum evidence package before calling the work research-grade:

1. Carry true ground-truth ETH labels through CSV and PCAP replay paths.
2. Remove file-global CAN features or recompute them causally within split-safe windows only.
3. Enforce train, validation, and test disjointness programmatically with overlap checks, not file-existence checks.
4. Eliminate split borrowing and heuristic ETH-to-CAN pairing in evaluation code.
5. Make the live runtime feature generation match the training representation, or retrain explicitly for the live representation.
6. Run CAN-only, ETH-only, and fused ablations under the same split policy.
7. Re-run cross-domain and LOAO evaluation with materially larger support and strict held-out protocols.
8. Update README claims to match the latest validated artifacts.

## Recommended Priority Order

1. Fix ETH runtime labels and ETH-only replay validity.
2. Remove leakage from CAN feature engineering and filename-derived ETH labels.
3. Harden split validation and stop borrowing training files into validation or test.
4. Align live deployment features with training features.
5. Re-run baselines, ablations, and generalization studies on clean splits.
6. Rewrite headline claims after the rerun, not before.

## Bottom Line

This repository is close to becoming a serious research platform, but it is not yet a reliable basis for strong performance claims. The core lesson from the current audit is simple:

- latency evidence looks credible
- detection evidence is not yet trustworthy
- hybrid CAN plus Ethernet claims are not yet validated end-to-end
