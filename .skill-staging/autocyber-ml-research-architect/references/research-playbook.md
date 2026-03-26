# Automotive Cyber-ML Research Playbook

## Scope

Use this reference when a task needs deeper domain guidance than the main skill body should carry.

## Map the problem to the system

- CAN or CAN-FD IDS: model frame IDs, payload bytes, timing deltas, burstiness, periodicity violations, and cross-ID dependencies.
- LIN or FlexRay analysis: emphasize schedule-aware anomalies, slot misuse, and timing consistency.
- Automotive Ethernet or IVI telemetry: expect richer packet metadata, service behavior, and lateral movement indicators.
- V2X security: consider misbehavior detection, message authenticity assumptions, PKI dependency, replay, spoofing, Sybil behavior, and communication loss.
- Autonomous driving stacks: separate attacks on perception, localization, planning, and sensor fusion; do not treat all sensor attacks as network IDS problems.

## Match the model family to the threat and hardware budget

- Lightweight sequence models: use GRU, LSTM, TCN, 1D CNN, or compact transformer variants for in-vehicle sequential telemetry.
- Reconstruction or density methods: use autoencoders, VAEs, diffusion-lite variants, or probabilistic detectors for anomaly detection with scarce labels.
- Multimodal models: use cross-attention or late-fusion designs when combining CAN, ECU logs, and sensor features.
- Graph or relation-aware models: use them when ECU interactions, message graphs, or fleet topology matter more than isolated frames.
- Federated learning: use it for fleet learning when raw telemetry cannot be centralized; evaluate aggregation trust, client drift, and poisoning resilience.
- Reinforcement learning: use it sparingly for adaptive response or policy selection; keep the reward-design and safety-risk discussion explicit.

## Design the feature pipeline

- Preserve timing information unless there is a strong reason to discard it.
- Compare raw-byte, signal-level, and engineered statistical features through ablations instead of assuming one representation wins.
- Use windowing schemes that reflect attack duration and detection-delay requirements.
- Include drift checks across vehicle models, firmware versions, routes, weather, and driver behavior when relevant.

## Evaluate like a security system

- Report AUROC or AUPRC, but also report F1, MCC, false positives per hour, detection delay, and class-wise recall for rare attacks.
- Measure latency, peak memory, model size, and energy cost for edge deployment claims.
- Test cross-vehicle, cross-route, and cross-firmware generalization to avoid overfitting to a single bench setup.
- Compare against simple baselines such as rules, statistical detectors, and shallow ML, not only deep models.
- Include ablations for features, window sizes, architectures, and defense mechanisms.

## Stress-test the model

- Test data poisoning, label corruption, and compromised-client behavior for federated settings.
- Test evasion with replay, mimicry, packet injection, stealth timing shifts, and adversarial perturbations where applicable.
- Test robustness under packet loss, missing sensors, clock drift, and benign distribution shift.
- Consider uncertainty estimation, conformal prediction, or calibrated confidence when false positives are operationally expensive.

## Defend the pipeline

- Protect data provenance and labeling integrity.
- Separate training, validation, and test splits by time, vehicle, route, or device when leakage is possible.
- Use robust aggregation, anomaly screening, and client reputation for federated learning.
- Pair ML with deterministic guards for safety-critical fallback behavior.
- Document privacy, update, and rollback strategy if models are meant to be updated over the air.

## Anchor to standards

- Use ISO/SAE 21434 to frame threat analysis, risk assessment, and cybersecurity case thinking.
- Use UNECE WP.29 R155 and R156 when discussing cybersecurity management and secure update governance.
- Mention ISO 26262-adjacent safety implications when model errors could propagate into safety-relevant decisions.
- Treat standards as design constraints and evidence expectations, not as substitutes for experimental validation.

## Reusable response template

1. Define the system and threat model.
2. State the data sources, labels, and preprocessing pipeline.
3. Propose the model family with hardware-aware justification.
4. Define baselines, ablations, and metrics.
5. Define adversarial and robustness experiments.
6. Explain deployment placement, update strategy, and monitoring.
7. Close with limitations, residual risks, and next experiments.
