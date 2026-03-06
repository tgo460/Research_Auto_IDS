import time
import torch
import numpy as np
import joblib
import pandas as pd
import argparse
import sys
import os
import random
from typing import Tuple, List

# Adjust path to import local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ONNX Runtime for edge-compliant inference (PDF standard)
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. Falling back to PyTorch inference.")

from architecture_improved import TinyHybridStudent
from router_replica import ConfidenceRouter, RouterConfig
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD
try:
    from dataloader_correlated_replica import CorrelatedHybridVehicleDataset
    HAS_DATALOADER = True
except ImportError:
    HAS_DATALOADER = False
    print("Warning: dataloader_correlated_replica not found or failed to import.")

class RealTimeEngine:
    def __init__(self, light_model_path: str, heavy_model_path: str, threshold: float = 0.6, input_dim: int = 16,
                 onnx_model_path: str = None, use_onnx: bool = True):
        self.device = torch.device("cpu") # Real-time usually CPU focused unless GPU available
        self.use_onnx = False  # Will be set to True if ONNX session loads successfully
        self.onnx_session = None
        self.input_dim = input_dim
        print(f"Initializing Real-Time Engine on {self.device}...")
        
        # 1. Load Light Model — prefer ONNX Runtime for edge deployment (PDF compliance)
        onnx_path = onnx_model_path or light_model_path.replace(".pth", ".onnx")
        
        if use_onnx and HAS_ONNX and os.path.exists(onnx_path):
            print(f"Loading Light Model (ONNX Runtime): {onnx_path}")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1  # Deterministic for real-time
            self.onnx_session = ort.InferenceSession(onnx_path, sess_options)
            self.onnx_input_names = [inp.name for inp in self.onnx_session.get_inputs()]
            self._validate_onnx_contract()
            self.use_onnx = True
            print(f"  ONNX inputs: {self.onnx_input_names}")
        else:
            if use_onnx and not HAS_ONNX:
                print("  onnxruntime not available, falling back to PyTorch.")
            elif use_onnx and not os.path.exists(onnx_path):
                print(f"  ONNX model not found at {onnx_path}, falling back to PyTorch.")
            print(f"Loading Light Model (PyTorch): {light_model_path}")
        
        # Always load PyTorch model as fallback
        self.light_model = TinyHybridStudent(input_dim=input_dim, hidden_dim=64, num_classes=2)
        if not os.path.exists(light_model_path):
            raise FileNotFoundError(f"Light model not found at {light_model_path}")
        checkpoint = torch.load(light_model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.light_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.light_model.load_state_dict(checkpoint)
        self.light_model.to(self.device)
        self.light_model.eval()
        
        inference_mode = "ONNX Runtime" if self.use_onnx else "PyTorch"
        print(f"  Light model inference backend: {inference_mode}")
        
        # 2. Load Heavy Model
        print(f"Loading Heavy Model: {heavy_model_path}")
        if not os.path.exists(heavy_model_path):
            raise FileNotFoundError(f"Heavy model not found at {heavy_model_path}")
        self.heavy_model = joblib.load(heavy_model_path)
        
        # 3. Configure Router
        print(f"Configuring Router with threshold: {threshold}")
        self.router = ConfidenceRouter(RouterConfig(threshold=threshold, route_if_below_or_equal=True))
        
        print("Engine Ready.\n")

    def _validate_onnx_contract(self) -> None:
        if self.onnx_session is None:
            return
        inputs = self.onnx_session.get_inputs()
        if len(inputs) != 2:
            raise ValueError(
                f"ONNX model must expose exactly 2 inputs (CAN, ETH). Found: {len(inputs)}"
            )
        can_shape = inputs[0].shape
        eth_shape = inputs[1].shape
        if len(can_shape) != 3:
            raise ValueError(f"CAN input rank must be 3, got shape: {can_shape}")
        if len(eth_shape) != 4:
            raise ValueError(f"ETH input rank must be 4, got shape: {eth_shape}")
        can_window = can_shape[1]
        can_channels = can_shape[2]
        eth_channels = eth_shape[1]
        if isinstance(can_window, int) and can_window != CAN_WINDOW_SIZE_STANDARD:
            raise ValueError(
                f"ONNX CAN window must be {CAN_WINDOW_SIZE_STANDARD}, got {can_window}"
            )
        if isinstance(can_channels, int) and can_channels != self.input_dim:
            raise ValueError(
                f"ONNX CAN channels must match input_dim={self.input_dim}, got {can_channels}"
            )
        if isinstance(eth_channels, int) and eth_channels != ETH_WINDOW_SIZE_STANDARD:
            raise ValueError(
                f"ONNX ETH channels must be {ETH_WINDOW_SIZE_STANDARD}, got {eth_channels}"
            )

    def process_packet(self, can_frame: torch.Tensor, eth_frame: torch.Tensor) -> dict:
        """
        Simulate processing a single packet pair (CAN + Ethernet).
        Returns a dictionary with result details.
        """
        start_time = time.time()
        
        # Ensure correct shape (Batch size 1)
        if can_frame.dim() == 2: # (L, C)
            can_frame = can_frame.unsqueeze(0) # (1, L, C)
        if eth_frame.dim() == 4: # (T, C, H, W)
            eth_frame = eth_frame.unsqueeze(0)
            
        can_frame = can_frame.to(self.device)
        eth_frame = eth_frame.to(self.device)
        if can_frame.dim() != 3:
            raise ValueError(f"CAN tensor must be rank-3 (B,L,C), got {tuple(can_frame.shape)}")
        if eth_frame.dim() not in (4, 5):
            raise ValueError(f"ETH tensor must be rank-4/5, got {tuple(eth_frame.shape)}")
        if can_frame.size(1) != CAN_WINDOW_SIZE_STANDARD:
            raise ValueError(
                f"CAN window size must be {CAN_WINDOW_SIZE_STANDARD}, got {can_frame.size(1)}"
            )
        
        # --- Stage 1: Light Model (ONNX Runtime or PyTorch) ---
        if self.use_onnx and self.onnx_session is not None:
            # Edge-compliant ONNX Runtime inference (PDF standard)
            try:
                can_np = can_frame.cpu().numpy()
                eth_np = eth_frame.cpu().numpy()
                # ONNX model expects 4D ETH input (B, C, H, W); squeeze temporal dim if 5D
                while eth_np.ndim > 4:
                    eth_np = eth_np.squeeze(1) if eth_np.shape[1] == 1 else eth_np.reshape(eth_np.shape[0], -1, eth_np.shape[-2], eth_np.shape[-1])
                feeds = {self.onnx_input_names[0]: can_np, self.onnx_input_names[1]: eth_np}
                ort_out = self.onnx_session.run(None, feeds)
                logits = torch.tensor(ort_out[0], device=self.device)
                probs = torch.softmax(logits, dim=1)
                confidence, light_pred = probs.max(dim=1)
            except Exception as e:
                print(f"ONNX inference failed, switching to PyTorch fallback: {e}")
                self.use_onnx = False
                with torch.no_grad():
                    logits = self.light_model(can_frame, eth_frame)
                    probs = torch.softmax(logits, dim=1)
                    confidence, light_pred = probs.max(dim=1)
        else:
            with torch.no_grad():
                logits = self.light_model(can_frame, eth_frame)
                probs = torch.softmax(logits, dim=1)
                confidence, light_pred = probs.max(dim=1)
            
        confidence_val = confidence.item()
        light_pred_val = light_pred.item()
        
        # --- Stage 2: Routing Decision ---
        should_route = self.router.route_from_confidence(confidence)
        
        final_pred = light_pred_val
        source = "LIGHT"
        heavy_time = 0.0
        
        if should_route.item():
            # --- Stage 3: Heavy Model (if routed) ---
            t0_heavy = time.time()
            source = "HEAVY"
            
            # Prepare input for Heavy Model (Concatenate flattened features)
            # CAN: (1, L, C) -> (1, L*C)
            # Check how many features the heavy model expects
            expected_features = getattr(self.heavy_model, 'n_features_in_', 1344)
            
            if expected_features == 1344 and can_frame.size(2) > 10:
                # Legacy heavy model (10 CAN features)
                can_frame_heavy = can_frame[:, :, :10]
            else:
                # Improved heavy model (16 CAN features) or matching dimensions
                can_frame_heavy = can_frame
                
            can_flat = can_frame_heavy.reshape(can_frame_heavy.size(0), -1).cpu().numpy()
            # ETH: (1, T, C, H, W) -> (1, T*C*H*W)
            eth_flat = eth_frame.reshape(eth_frame.size(0), -1).cpu().numpy()
            
            heavy_input = np.concatenate([can_flat, eth_flat], axis=1)
            
            # Predict
            heavy_pred = self.heavy_model.predict(heavy_input)[0]
            final_pred = heavy_pred
            heavy_time = (time.time() - t0_heavy) * 1000 # ms
            
        total_time = (time.time() - start_time) * 1000 # ms
        
        return {
            "source": source,
            "prediction": final_pred,
            "confidence": confidence_val,
            "latency_ms": total_time,
            "heavy_overhead_ms": heavy_time
        }

def simulate_stream(engine, dataset, limit=100, delay=0.0):
    """
    Simulates a live stream by iterating through the dataset.
    """
    print(f"{'ID':<6} | {'SOURCE':<8} | {'CONF':<6} | {'PRED':<10} | {'LATENCY (ms)':<12} | {'STATUS'}")
    print("-" * 75)
    
    malicious_count = 0
    routed_count = 0
    total_latency = 0
    correct_count = 0
    
    indices = list(range(len(dataset)))
    # Shuffle indices to get a mix for the demo stream
    random.shuffle(indices)
    indices = indices[:limit]
    
    for i, idx in enumerate(indices):
        data = dataset[idx]
        if isinstance(data, tuple):
             if len(data) == 2:
                 (xc, xe), label = data
             else:
                 xc, xe, label = data
        elif isinstance(data, dict):
             xc = data['can']
             xe = data['eth']
             label = data['label']
        else:
             print(f"Unknown data type: {type(data)}")
             continue
             
        if isinstance(label, torch.Tensor):
            label = int(label.item())

        # Mock label if not provided or structure is different
        if not isinstance(label, (int, np.integer)):
             label = 0 

        res = engine.process_packet(xc, xe)
        
        # Output Formatting
        pred_label = "MALICIOUS" if res['prediction'] == 1 else "NORMAL"
        actual_label = "MALICIOUS" if label == 1 else "NORMAL"
        status = "CORRECT" if res['prediction'] == label else f"MISSED (Actual: {label})"
        
        if res['prediction'] == label:
            correct_count += 1
        
        # Color coding (simulation)
        source_str = res['source']
        if source_str == "HEAVY":
            routed_count += 1
            
        print(f"{i:<6} | {source_str:<8} | {res['confidence']:.4f} | {pred_label:<10} | {res['latency_ms']:<12.2f} | {status}")
        
        total_latency += res['latency_ms']
        
        if delay > 0:
            time.sleep(delay)
            
    avg_latency = total_latency / limit
    accuracy = correct_count / limit
    print("-" * 75)
    print(f"Stream Complete.")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Routed Packets: {routed_count}/{limit} ({routed_count/limit:.1%})")
    print(f"Accuracy: {accuracy:.1%}")

def main():
    parser = argparse.ArgumentParser(description="Run Real-Time Inference Simulation")
    parser.add_argument("--light_model", type=str, default="models/student_tiny_improved.pth", help="Path to light model")
    parser.add_argument("--heavy_model", type=str, default="models/heavy_rf_improved.joblib", help="Path to heavy model")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Data directory")
    parser.add_argument("--limit", type=int, default=50, help="Number of packets to simulate")
    parser.add_argument("--delay", type=float, default=0.05, help="Artificial delay between packets (sec)")
    parser.add_argument("--threshold", type=float, default=0.5781, help="Router threshold")
    parser.add_argument("--use_onnx", action="store_true", default=True, help="Use ONNX Runtime for light model (edge deployment)")
    parser.add_argument("--no_onnx", dest="use_onnx", action="store_false", help="Disable ONNX Runtime, use PyTorch")
    args = parser.parse_args()
    
    # Check if we are using improved model (16 engineered features)
    is_improved = "improved" in args.light_model
    input_dim = 16 if is_improved else 10
    
    # Initialize Engine
    try:
        engine = RealTimeEngine(args.light_model, args.heavy_model, args.threshold,
                                input_dim=input_dim, use_onnx=args.use_onnx)
    except Exception as e:
        print(f"Error initializing engine: {e}")
        return

    print("Loading Validation Dataset for Simulation...")
    # Heuristic based on workspace_info
    # We'll use a known pair
    can_file = "can_fuzzy_train.csv"
    eth_npy = "eth_driving_02_injected_images-008.npy"
    eth_csv_base = "eth_driving_02_injected" 
    
    # Point to engineered CAN if using improved model
    if is_improved:
        engineered_dir = os.path.join(args.data_dir, "replica_can_b1_engineered")
        if os.path.exists(os.path.join(engineered_dir, can_file)):
            can_path = os.path.join(engineered_dir, can_file)
            print(f"Using engineered CAN file: {can_path}")
        else:
            can_path = os.path.join(args.data_dir, can_file)
    else:
        can_path = os.path.join(args.data_dir, can_file)
        
    eth_npy_path = os.path.join(args.data_dir, eth_npy)
    
    # Prioritize finding the correct ETH CSV with timestamps
    possible_eth_csvs = [
        os.path.join(args.data_dir, "replica_eth_smoke", f"{eth_csv_base}_replica_packets.csv"),
        os.path.join(args.data_dir, f"{eth_csv_base}_replica_packets.csv"),
        os.path.join(args.data_dir, f"{eth_csv_base}_preprocessed.csv"),
        os.path.join(args.data_dir, f"{eth_csv_base}.csv")
    ]
    
    eth_csv_path = None
    for p in possible_eth_csvs:
        if os.path.exists(p):
            # Quick check for timestamp
            try:
                # Read just header
                df_head = pd.read_csv(p, nrows=1)
                if 'timestamp_sec' in df_head.columns:
                    eth_csv_path = p
                    print(f"Found suitable ETH CSV with timestamps: {p}")
                    break
                else:
                    print(f"Skipping {p}: Missing 'timestamp_sec'")
            except:
                pass
                
    if not eth_csv_path and os.path.exists(possible_eth_csvs[-1]):
         # Fallback to the basic one if nothing else, but it might fail later
         eth_csv_path = possible_eth_csvs[-1]
         print(f"Warning: Using fallback ETH CSV {eth_csv_path} (might lack timestamps)")

    dataset = None
    if HAS_DATALOADER:
        try:
            # Check necessary files
            if os.path.exists(can_path) and os.path.exists(eth_csv_path) and os.path.exists(eth_npy_path):
                # Features
                if is_improved:
                    can_features = ['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
                                    'can_id_freq_global', 'can_id_freq_win', 'payload_entropy', 
                                    'inter_arrival', 'inter_arrival_roll_mean', 'id_switch_rate_win']
                else:
                    can_features=['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
                    
                dataset = CorrelatedHybridVehicleDataset(
                    can_csv_path=can_path,
                    eth_packet_csv_path=eth_csv_path,
                    eth_npy_path=eth_npy_path,
                    # Must fix feature columns
                    can_features=can_features,
                    can_window_size=CAN_WINDOW_SIZE_STANDARD,
                    eth_window_size=ETH_WINDOW_SIZE_STANDARD,
                    eth_overlap=0
                )
                print(f"Loaded {len(dataset)} samples from real data.")
            else:
                print("Missing real data files for specific set. Checking alternate...")
                if not os.path.exists(can_path): print(f"Missing {can_path}")
                if not os.path.exists(eth_csv_path): print(f"Missing {eth_csv_path}")
        except Exception as e:
            print(f"Failed to load real dataset: {e}")
            
    if dataset is None or len(dataset) == 0:
        print("Using Synthetic Dummy Data Generator as fallback...")
        class DummyDataset:
            def __init__(self, size, input_dim=10):
                self.size = size
                self.input_dim = input_dim
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                # (32, input_dim), (1, 1, 64, 64) approx
                # Must match types
                xc = torch.randn(100, self.input_dim)
                xe = torch.randn(1, 1, 32, 32)
                label = random.randint(0, 1)
                return (xc, xe), label
        
        dataset = DummyDataset(args.limit * 2, input_dim=input_dim)

    simulate_stream(engine, dataset, limit=args.limit, delay=args.delay)

if __name__ == "__main__":
    main()
