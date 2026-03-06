import numpy as np
import onnxruntime as ort
import time
import os


def _dim_or_default(dim, default: int) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    try:
        # ONNX Runtime may return strings for symbolic dimensions.
        v = int(dim)
        return v if v > 0 else default
    except Exception:
        return default


def test_onnx_model(onnx_path="models/student_tiny_improved.onnx"):
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return

    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    
    # Get input names and shapes
    inputs_info = session.get_inputs()
    input_names = [inp.name for inp in inputs_info]
    print(f"Model inputs: {input_names}")
    
    for inp in inputs_info:
        print(f"  - {inp.name}: shape {inp.shape}, type {inp.type}")
    
    if len(inputs_info) != 2:
        raise RuntimeError(f"Expected 2 inputs (CAN, ETH), got {len(inputs_info)}")

    can_info, eth_info = inputs_info[0], inputs_info[1]
    can_shape = can_info.shape
    eth_shape = eth_info.shape

    if len(can_shape) != 3:
        raise RuntimeError(f"CAN input must be rank-3 (B,L,C), got {can_shape}")
    if len(eth_shape) != 4:
        raise RuntimeError(f"ETH input must be rank-4 (B,C,H,W), got {eth_shape}")

    b = _dim_or_default(can_shape[0], 1)
    can_l = _dim_or_default(can_shape[1], 100)
    can_c = _dim_or_default(can_shape[2], 16)
    eth_c = _dim_or_default(eth_shape[1], 1)
    eth_h = _dim_or_default(eth_shape[2], 32)
    eth_w = _dim_or_default(eth_shape[3], 32)

    # Create dummy data that exactly matches exported input contract.
    dummy_can = np.random.randn(b, can_l, can_c).astype(np.float32)
    dummy_eth = np.random.randn(b, eth_c, eth_h, eth_w).astype(np.float32)
    
    inputs = {
        input_names[0]: dummy_can,
        input_names[1]: dummy_eth
    }
    
    print("\nRunning inference...")
    
    # Warmup
    for _ in range(5):
        session.run(None, inputs)
        
    # Measure time
    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()
    
    print(f"Inference successful!")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output values (logits): {outputs[0]}")
    
    # Convert logits to probabilities
    exp_logits = np.exp(outputs[0] - np.max(outputs[0], axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    print(f"Output probabilities: {probs}")
    
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    test_onnx_model()
