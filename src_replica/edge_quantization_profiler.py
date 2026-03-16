import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from architecture_improved import TinyHybridStudent
from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD
from src_replica.loao_train_replica import CAN_FEATURES_16, _load_pair

def measure_latency(model, dataloader, device, num_warmup=10):
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_warmup:
                break
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            _ = model(b_can, b_eth)

    # Formal Measurement
    with torch.no_grad():
        for batch in dataloader:
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            
            start_time = time.perf_counter()
            _ = model(b_can, b_eth)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000) # Convert to ms
            
    return np.mean(latencies), np.std(latencies), np.percentile(latencies, 95)

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            b_can = batch["can"].to(device)
            b_eth = batch["eth"].to(device)
            b_y = batch["label"].to(device)
            
            out = model(b_can, b_eth)
            preds = torch.argmax(out, dim=1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
            
    return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the FP32 model")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=1, help="Simulate real-time streaming batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device for profiling (usually CPU for edge constraints)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"--- Edge Deployment Profiling (Targeting: {device}) ---")
    
    # Load sample dataset
    # We just need any valid domain to run standard inference loops
    print("Loading benchmark dataset...")
    ds = _load_pair(
        args.data_dir, 
        "can_dos_train.csv", 
        "eth_driving_01_injected_images-003.npy", 
        max_rows=500
    )
    if ds is None:
        print("Failed to load dataset, make sure files exist.")
        return
        
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # Init and Load FP32 Base Model
    model_fp32 = TinyHybridStudent(
        input_dim=len(CAN_FEATURES_16),
        hidden_dim=32,
        num_classes=2
    ).to(device)
    
    if os.path.exists(args.model_path):
        model_fp32.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded FP32 Model from {args.model_path}")
    else:
        print(f"Base model not found: {args.model_path}")
        return
        
    # --- Profiling FP32 ---
    print("\nProfiling FP32 Base Model...")
    fp32_acc = evaluate_accuracy(model_fp32, loader, device)
    fp_mean, fp_std, fp_p95 = measure_latency(model_fp32, loader, device)
    
    print(f"  Accuracy: {fp32_acc*100:.2f}%")
    print(f"  Latency (Batch {args.batch_size}): {fp_mean:.2f}ms ± {fp_std:.2f}ms (p95: {fp_p95:.2f}ms)")
    
    # --- Quantization INT8 ---
    # Apply Dynamic Quantization to Linear and Conv1d layers (common in TCN architecture)
    print("\nApplying Dynamic Post-Training Quantization (INT8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear, nn.Conv1d, nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # --- Profiling INT8 ---
    print("Profiling Quantized INT8 Model...")
    int8_acc = evaluate_accuracy(quantized_model, loader, device)
    int8_mean, int8_std, int8_p95 = measure_latency(quantized_model, loader, device)
    
    print(f"  Accuracy: {int8_acc*100:.2f}%")
    print(f"  Latency (Batch {args.batch_size}): {int8_mean:.2f}ms ± {int8_std:.2f}ms (p95: {int8_p95:.2f}ms)")
    
    acc_drop = fp32_acc - int8_acc
    speedup = fp_mean / int8_mean if int8_mean > 0 else 0
    print(f"\n--- Output Summary ---")
    print(f"Accuracy Drop (FP32 -> INT8): {acc_drop*100:.2f}%")
    print(f"Latency Speedup: {speedup:.2f}x")
    
    report = {
        "timestamp": time.time(),
        "batch_size": args.batch_size,
        "device": str(device),
        "fp32": {
            "accuracy": float(fp32_acc),
            "latency_mean_ms": float(fp_mean),
            "latency_std_ms": float(fp_std),
            "latency_p95_ms": float(fp_p95),
            "model_size_mb": os.path.getsize(args.model_path) / (1024 * 1024)
        },
        "int8_dynamic": {
            "accuracy": float(int8_acc),
            "latency_mean_ms": float(int8_mean),
            "latency_std_ms": float(int8_std),
            "latency_p95_ms": float(int8_p95),
            "speedup_ratio": float(speedup)
        }
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "edge_quantization_report_replica.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()