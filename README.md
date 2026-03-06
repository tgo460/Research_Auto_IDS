# Research_Auto_IDS

**A Lightweight Hybrid Intrusion Detection System for Automotive CAN + Ethernet Networks**

A research-grade, deployment-ready IDS that fuses CAN bus telemetry with Ethernet packet imagery through a two-tier cascade architecture (fast light model → heavy fallback), achieving real-time detection under automotive timing constraints (< 100 ms latency) on resource-constrained ECUs.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Quick Start — Training Pipeline](#quick-start--training-pipeline)
7. [Real-Time Inference Simulation](#real-time-inference-simulation)
8. [ONNX Edge Deployment](#onnx-edge-deployment)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Explainability (SHAP)](#explainability-shap)
11. [Runtime CLI Commands](#runtime-cli-commands)
12. [Compliance & Safety Artifacts](#compliance--safety-artifacts)
13. [Key Configuration Parameters](#key-configuration-parameters)
14. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Automotive In-Vehicle Network                  │
│                                                                 │
│   CAN Bus (Messages)              Ethernet (Packets/Images)    │
│        │                                    │                   │
│        ▼                                    ▼                   │
│   ┌──────────────┐               ┌───────────────────┐         │
│   │ 16 Engineered│               │ Image Embeddings  │         │
│   │ CAN Features │               │ (.npy 32×32 frames)│        │
│   └──────┬───────┘               └────────┬──────────┘         │
│          │                                │                     │
│          ▼                                ▼                     │
│   ┌──────────────┐               ┌───────────────────┐         │
│   │ TCN Branch   │               │  CNN Branch       │         │
│   │ (DepthSep    │               │  (Conv2d →        │         │
│   │  1D Conv)    │               │   AdaptivePool)   │         │
│   └──────┬───────┘               └────────┬──────────┘         │
│          │                                │                     │
│          └───────────┬────────────────────┘                     │
│                      ▼                                          │
│           ┌─────────────────┐                                   │
│           │  Hybrid Fusion  │                                   │
│           │ (Concat → MLP)  │                                   │
│           └────────┬────────┘                                   │
│                    ▼                                            │
│           ┌─────────────────┐    Confidence     ┌────────────┐ │
│           │  Light Model    │───── < θ ────────▶│Heavy Model │ │
│           │  (TinyHybrid    │   (ConfRouter)    │(Random     │ │
│           │   Student)      │                   │ Forest)    │ │
│           └────────┬────────┘                   └─────┬──────┘ │
│                    │                                  │         │
│                    └──────────┬────────────────────────┘         │
│                               ▼                                 │
│                    ┌─────────────────┐                           │
│                    │  NORMAL  or     │                           │
│                    │  MALICIOUS      │                           │
│                    └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

**Two-Tier Cascade Pipeline:**

| Tier | Model | Backend | Input | Latency | Purpose |
|------|-------|---------|-------|---------|---------|
| Light | `TinyHybridStudent` | ONNX Runtime / PyTorch | 100 CAN msgs + 1 ETH frame | ~0.4 ms | Fast triage |
| Heavy | `RandomForestClassifier` | scikit-learn | 2624 flattened features | ~17 ms | High-accuracy fallback |

The **Confidence Router** sends uncertain predictions (softmax confidence ≤ threshold) to the heavy model. Typical routing rate: 3–5% of traffic.

---

## Project Structure

```
Research_Auto_IDS/
├── README.md                          ← This file
├── PLAN.md                            ← Gap-closure checklist (C01–C18)
├── compliance_matrix.md               ← Traceability matrix
│
├── configs/
│   ├── deployment.example.json        ← Reference deployment config
│   ├── deployment.schema.json         ← JSON schema for config validation
│   ├── benchmark_report.schema.json   ← Benchmark output schema
│   └── repro_full_and_v2.json         ← Reproducibility pipeline
│
├── datasets/
│   ├── can_dos_train.csv              ← CAN attack datasets (DoS, Fuzzy,
│   ├── can_fuzzy_train.csv               Gear spoofing, RPM spoofing, Normal)
│   ├── can_gear_train.csv
│   ├── can_rpm_train.csv
│   ├── can_normal_train.csv
│   ├── eth_driving_*_injected.csv     ← Ethernet injected/original pairs
│   ├── eth_driving_*_original.csv
│   ├── eth_*_images*.npy              ← Ethernet image embeddings (32×32)
│   ├── replica_can_b1_engineered/     ← Engineered CAN features (16 cols)
│   ├── replica_can_b1_baseline/       ← Raw CAN baseline
│   ├── replica_eth_smoke/             ← Ethernet with timestamps
│   ├── replica_correlation/           ← CAN↔ETH alignment test data
│   ├── Car-Hacking Dataset/           ← Source CAN attack data
│   └── autoeth-intrusion-dataset/     ← Source Ethernet intrusion data
│
├── models/
│   ├── student_tiny_improved.pth      ← Light model (PyTorch)
│   ├── student_tiny_improved.onnx     ← Light model (ONNX, edge-ready)
│   ├── heavy_rf_improved.joblib       ← Heavy model (Random Forest)
│   └── checkpoints/                   ← Training checkpoints
│
├── reports/
│   ├── shap_summary_improved.png      ← SHAP feature importance plot
│   ├── confusion_*.png                ← Confusion matrices
│   ├── latency_summary.png            ← Latency profiling
│   ├── cascade_eval_report_*.json     ← Cascade evaluation reports
│   └── ...                            ← 200+ evaluation artifacts
│
├── logs/                              ← Timestamped training/eval logs
│
├── src_replica/                       ← Core source code
│   ├── architecture_improved.py       ← TinyHybridStudent model definition
│   ├── features_can_replica.py        ← CAN feature engineering pipeline
│   ├── features_eth_replica.py        ← Ethernet feature extraction
│   ├── correlation_replica.py         ← CAN↔ETH timestamp alignment
│   ├── dataloader_correlated_replica.py ← Aligned dataset loader
│   ├── router_replica.py              ← Confidence-based routing
│   ├── train_improved_light_model.py  ← Train light model (class-weighted)
│   ├── train_heavy_model_improved.py  ← Train heavy Random Forest
│   ├── realtime_inference_replica.py  ← Real-time inference engine
│   ├── export_onnx_replica.py         ← ONNX export for edge deployment
│   ├── test_onnx_replica.py           ← ONNX model validation
│   ├── quantize_onnx_replica.py       ← INT8/FP16 ONNX quantization
│   ├── explainability_replica.py      ← SHAP explainability analysis
│   ├── evaluate_improved_replica.py   ← Model evaluation with metrics
│   ├── cascade_eval_replica.py        ← Full cascade pipeline evaluation
│   ├── robustness_eval_replica.py     ← Adversarial robustness testing
│   ├── coordinated_attack_replay.py   ← Cross-network attack simulation
│   └── runtime/                       ← Production runtime modules
│       ├── engine.py                  ← Core inference engine
│       ├── adapters.py                ← CAN/Ethernet I/O adapters
│       ├── standards.py               ← Window size constants
│       ├── contracts.py               ← AlertRecord, BenchmarkReport
│       ├── config.py                  ← Configuration management
│       ├── security.py                ← SHA256 model integrity checks
│       └── metrics.py                 ← Latency/deadline tracking
│
├── docs/
│   └── safety_case.md                 ← ISO 26262 safety case
│
└── *.mermaid                          ← Architecture diagrams
```

---

## Prerequisites

- **Python** 3.10+ (tested on 3.13)
- **Operating System**: Windows 10/11 or Linux
- **RAM**: 8 GB minimum (16 GB recommended for full dataset training)
- **GPU**: Optional (CPU inference achieves < 1 ms latency)

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Research_Auto_IDS

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1
# Activate (Linux/macOS)
# source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# (Optional) For ONNX export
pip install onnx onnxscript

# (Optional) For live PCAP/CAN interfaces
pip install scapy python-can
```

> **Windows Long Path Note**: If `pip install onnx` fails with `[WinError 206]`, install to a shorter path:
> ```bash
> mkdir C:\onnx_pkg
> pip install onnx onnxscript --target C:\onnx_pkg
> ```
> The export script already includes `sys.path.insert(0, r"C:\onnx_pkg")`.

---

## Dataset Preparation

### Source Datasets

1. **CAN Bus**: [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) (KIA SOUL — DoS, Fuzzy, Gear Spoofing, RPM Spoofing, Normal)
2. **Ethernet**: AutoETH Intrusion Dataset (driving/indoor scenarios with injected attacks)

### Feature Engineering

The CAN feature pipeline (`features_can_replica.py`) produces 16 engineered features from raw CAN messages:

| # | Feature | Description |
|---|---------|-------------|
| 1–8 | `CAN_ID`, `DLC`, `D0`–`D5` | Raw CAN fields |
| 9–10 | `D6`, `D7` | Payload bytes |
| 11 | `can_id_freq_global` | Global CAN ID frequency distribution |
| 12 | `can_id_freq_win` | Windowed CAN ID frequency |
| 13 | `payload_entropy` | Shannon entropy of payload bytes |
| 14 | `inter_arrival` | Time between consecutive CAN messages |
| 15 | `inter_arrival_roll_mean` | Rolling mean of inter-arrival times |
| 16 | `id_switch_rate_win` | Rate of CAN ID changes in window |

Pre-engineered CSVs are stored in `datasets/replica_can_b1_engineered/`.

Ethernet data uses 32×32 grayscale image embeddings stored as `.npy` files, with timestamp-aligned CSVs in `datasets/replica_eth_smoke/`.

---

## Quick Start — Training Pipeline

### Step 1: Train the Light Model

```bash
python src_replica/train_improved_light_model.py \
    --data_dir datasets \
    --output_dir models \
    --epochs 5 \
    --batch_size 32 \
    --lr 0.001
```

This trains the `TinyHybridStudent` model with:
- **Class-weighted loss**: Automatically computes weights from label distribution to handle imbalanced attack/normal ratios
- **100-message CAN windows**: Per automotive IDS literature standards
- **ReduceLROnPlateau** scheduler tracking validation F1

Output: `models/student_tiny_improved.pth`

### Step 2: Train the Heavy Model

```bash
python src_replica/train_heavy_model_improved.py \
    --data_dir datasets \
    --output_dir models \
    --max_rows 5000
```

Trains a `RandomForestClassifier` (100 trees) on flattened CAN+ETH features (2624 dimensions = 100×16 CAN + 1024 ETH).

Output: `models/heavy_rf_improved.joblib`

### Step 3: Export to ONNX (Edge Deployment)

```bash
python src_replica/export_onnx_replica.py \
    --model_path models/student_tiny_improved.pth \
    --output_path models/student_tiny_improved.onnx \
    --input_dim 16
```

Exports the light model to ONNX format for deployment on automotive ECUs via ONNX Runtime, TensorRT, or OpenVINO.

Output: `models/student_tiny_improved.onnx`

---

## Real-Time Inference Simulation

```bash
# Run with ONNX Runtime backend (default — edge-compliant)
python src_replica/realtime_inference_replica.py \
    --limit 100 \
    --threshold 0.99 \
    --use_onnx

# Run with PyTorch backend (fallback)
python src_replica/realtime_inference_replica.py \
    --limit 100 \
    --threshold 0.5781 \
    --no_onnx
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--light_model` | `models/student_tiny_improved.pth` | Light model checkpoint |
| `--heavy_model` | `models/heavy_rf_improved.joblib` | Heavy model path |
| `--threshold` | `0.5781` | Confidence router threshold |
| `--limit` | `50` | Number of packets to simulate |
| `--delay` | `0.05` | Inter-packet delay (seconds) |
| `--use_onnx` | `True` | Use ONNX Runtime for light model |
| `--no_onnx` | — | Force PyTorch inference |

### Expected Output

```
ID     | SOURCE   | CONF   | PRED       | LATENCY (ms) | STATUS
---------------------------------------------------------------------------
0      | LIGHT    | 0.9999 | MALICIOUS  | 0.66         | CORRECT
1      | LIGHT    | 0.9988 | MALICIOUS  | 0.43         | CORRECT
...
9      | HEAVY    | 0.9694 | MALICIOUS  | 17.89        | CORRECT
---------------------------------------------------------------------------
Stream Complete.
Avg Latency: 1.08 ms
Routed Packets: 1/30 (3.3%)
Accuracy: 100.0%
```

---

## ONNX Edge Deployment

### Validate the ONNX Model

```bash
python src_replica/test_onnx_replica.py
```

Expected output:
```
Model inputs: ['can_input', 'eth_input']
  - can_input: shape ['batch_size', 100, 16], type tensor(float)
  - eth_input: shape ['batch_size', 1, 32, 32], type tensor(float)
Inference Time: ~0.21 ms
```

### Quantize for ECU Deployment (Optional)

```bash
python src_replica/quantize_onnx_replica.py
```

Produces INT8/FP16 quantized models for further latency reduction on ARM-based ECUs.

### Integration with Automotive Middleware

The ONNX model can be deployed using:
- **ONNX Runtime C++ API** on automotive ECUs (ARM Cortex-A/M)
- **TensorRT** for NVIDIA Jetson platforms
- **OpenVINO** for Intel-based compute modules
- **AUTOSAR / ROS 2** middleware integration via `src_replica/runtime/adapters.py`

---

## Evaluation & Metrics

### Evaluate the Light Model

```bash
python src_replica/evaluate_improved_replica.py --data_dir datasets
```

Reports: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

### Evaluate the Full Cascade

```bash
python src_replica/cascade_eval_replica.py --data_dir datasets
```

Tests the complete Light → Router → Heavy pipeline with per-tier metrics.

### Robustness Testing

```bash
python src_replica/robustness_eval_replica.py
```

### Key Performance Results

| Metric | Value | Target (PDF Standard) |
|--------|-------|-----------------------|
| Detection Accuracy | 100% | > 95% |
| Average Latency | 1.08 ms | < 100 ms |
| ONNX Inference | 0.21 ms | < 100 ms |
| False Positive Rate | < 5% | < 5% |
| CAN Window Size | 100 messages | 100 messages |
| Heavy Model Routing | ~3% | Confidence-based |

---

## Explainability (SHAP)

```bash
python src_replica/explainability_replica.py
```

Generates a SHAP summary plot (`reports/shap_summary_improved.png`) showing which engineered features contribute most to the heavy model's attack detection decisions.

Key features typically driving detection:
- `payload_entropy` — Shannon entropy anomalies in CAN payloads
- `inter_arrival` — Abnormal message timing (e.g., DoS flooding)
- `can_id_freq_global` — Unusual CAN ID distribution shifts
- `id_switch_rate_win` — Rapid ID switching in fuzzy attacks

---

## Runtime CLI Commands

Production-grade CLI for deployment and benchmarking:

```bash
# Start the IDS engine
python run_ids.py --config configs/deployment.example.json

# Run performance benchmark
python benchmark_ids.py --config configs/deployment.example.json \
    --output reports/benchmark_ids_report.json

# Validate deployment configuration
python validate_ids.py --config configs/deployment.example.json

# Replay recorded CAN+ETH traffic
python replay_can_eth.py --config configs/deployment.example.json \
    --can_csv datasets/can_dos_train.csv \
    --eth_csv datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv

# Full reproducibility pipeline
python reproduce.py --config configs/repro_full_and_v2.json
```

---

## Compliance & Safety Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Deployment Schema | `configs/deployment.schema.json` | JSON schema for runtime config validation |
| Benchmark Schema | `configs/benchmark_report.schema.json` | Standardized benchmark output format |
| Traceability Matrix | `compliance_matrix.md` | Maps requirements C01–C18 to artifacts |
| Safety Case | `docs/safety_case.md` | ISO 26262 functional safety documentation |
| Gap Checklist | `PLAN.md` | 18-item deployment readiness checklist |

---

## Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `can_window_size` | 100 | PDF standard: 100 CAN messages per detection window |
| `eth_window_size` | 50 (default) / 1 (single frame) | Configurable per deployment |
| `input_dim` | 16 | 16 engineered CAN features |
| `hidden_dim` | 64 | TCN hidden dimension |
| `num_classes` | 2 | Binary: Normal / Malicious |
| `router_threshold` | 0.5781 (tuned) / 0.99 (strict) | Confidence routing cutoff |
| `tolerance_ms` | 100.0 | CAN↔ETH timestamp alignment tolerance |

---

## Troubleshooting

### ONNX Install Fails on Windows (`WinError 206`)
The ONNX package has extremely long file paths in its test data. Solution:
```bash
mkdir C:\onnx_pkg
pip install onnx onnxscript --target C:\onnx_pkg
```
The export script automatically includes this path.

### Low Aligned Pairs Count
If `CorrelatedHybridVehicleDataset` reports few aligned pairs:
- Verify ETH CSV has a `timestamp_sec` column
- Use CSVs from `datasets/replica_eth_smoke/` (pre-processed with timestamps)
- Adjust `tolerance_ms` (default: 100 ms)

### Heavy Model Feature Mismatch
If the heavy model raises `ValueError: X has N features, but expecting M`:
- Retrain: `python src_replica/train_heavy_model_improved.py`
- The heavy model must be trained with the same `can_window_size` used for inference

### ONNX Dimension Errors
If ONNX Runtime reports shape mismatches:
- Ensure `export_onnx_replica.py` uses the same window/image dimensions as inference
- Current expected shapes: CAN `(B, 100, 16)`, ETH `(B, 1, 32, 32)`
- Re-export: `python src_replica/export_onnx_replica.py`

---

## Reproducing Results from Scratch

This section provides a complete, step-by-step guide for researchers to reproduce
our results starting from a fresh clone.

### Step 0: Clone and Install

```bash
git clone https://github.com/tgo460/Research_Auto_IDS.git
cd Research_Auto_IDS

python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 1: Obtain Datasets

This project uses two publicly available datasets (not included in the repo due to size):

| Dataset | Source | Size | Target Directory |
|---------|--------|------|------------------|
| Car-Hacking Dataset (KIA SOUL) | [OCSLAB](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) | ~1.2 GB | `datasets/Car-Hacking Dataset/` |
| AutoETH Intrusion Dataset | [Zenodo](https://zenodo.org/records/14643663) | ~2 GB | `datasets/autoeth-intrusion-dataset/` |

After downloading, run the automated setup script:

```bash
python setup_datasets.py
```

This will:
1. Verify datasets are present
2. Extract CAN training CSVs from the raw dataset files
3. Engineer 16 CAN features per row → `datasets/replica_can_b1_engineered/`
4. Extract Ethernet packet CSVs from PCAPs → `datasets/replica_eth_smoke/` (requires `pip install scapy`)
5. Verify all required files are in place

### Step 2: Train Models

```bash
# Train the light model (TinyHybridStudent — TCN+CNN fusion)
python src_replica/train_improved_light_model.py \
    --data_dir datasets --output_dir models --epochs 5 --batch_size 32 --lr 0.001

# Train the heavy model (Random Forest — 100 trees, 2624 features)
python src_replica/train_heavy_model_improved.py \
    --data_dir datasets --output_dir models

# Export light model to ONNX for edge deployment
python src_replica/export_onnx_replica.py \
    --model_path models/student_tiny_improved.pth \
    --output_path models/student_tiny_improved.onnx
```

### Step 3: Validate & Benchmark

```bash
# Validate deployment config and model integrity
python validate_ids.py --config configs/deployment.example.json

# Run performance benchmark
python benchmark_ids.py --config configs/deployment.example.json \
    --output reports/benchmark_ids_report.json
```

### Step 4: Evaluate

```bash
# Full cascade evaluation (Light → Router → Heavy)
python src_replica/cascade_eval_replica.py --data_dir datasets

# Aggregate final metrics with split validation
python evaluate.py --base-path . --out-dir reports --strict-split-check \
    --split-manifest data/splits/split_v2_domain_balanced.json
```

### Step 5: Run the Full Reproducibility Pipeline (One Command)

Alternatively, after Steps 0–2, run everything at once:

```bash
python reproduce.py --config configs/repro_full_and_v2.json
```

This executes validate → benchmark → evaluate in sequence using the canonical
configuration in `configs/repro_full_and_v2.json`.

### Expected Results

| Metric | Expected Value | Target |
|--------|---------------|--------|
| Detection Accuracy | 100% | > 95% |
| False Positive Rate | < 5% | < 5% |
| Average Latency | ~1 ms | < 100 ms |
| ONNX Inference Latency | ~0.2 ms | < 100 ms |
| Heavy Model Routing Rate | ~3% | Confidence-based |
| CAN Window Size | 100 messages | Standard |

Results are written to `reports/` as JSON files and PNG visualizations.

### Reproducibility Checklist

- [ ] Python 3.10+ installed
- [ ] `requirements.txt` dependencies installed
- [ ] Car-Hacking Dataset downloaded and placed in `datasets/Car-Hacking Dataset/`
- [ ] AutoETH Dataset downloaded and placed in `datasets/autoeth-intrusion-dataset/`
- [ ] `python setup_datasets.py` completed successfully
- [ ] Models trained (or pre-trained models present in `models/`)
- [ ] `python reproduce.py` exits with code 0
- [ ] `reports/final_metrics_latest.json` matches expected values

---

## Citation

If you use this project in your research, please cite:

```
@misc{research_auto_ids,
  title={Lightweight Hybrid Intrusion Detection System for Automotive CAN and Ethernet Networks},
  year={2026},
  note={Two-tier cascade architecture with ONNX edge deployment}
}
```

---

## License

This project is for academic research purposes.
