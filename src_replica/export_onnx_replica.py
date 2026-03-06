import os
import sys
sys.path.insert(0, r"C:\onnx_pkg")
import argparse
import torch
import torch.onnx
from architecture_improved import TinyHybridStudent
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD

def main():
    parser = argparse.ArgumentParser(description="Export Light Model to ONNX for Edge Deployment")
    parser.add_argument("--model_path", type=str, default="models/student_tiny_improved.pth", help="Path to PyTorch model")
    parser.add_argument("--output_path", type=str, default="models/student_tiny_improved.onnx", help="Output ONNX path")
    parser.add_argument("--input_dim", type=int, default=16, help="Number of CAN features")
    parser.add_argument("--can_window_size", type=int, default=CAN_WINDOW_SIZE_STANDARD, help="CAN window size")
    args = parser.parse_args()

    print(f"Loading PyTorch model from {args.model_path}...")
    device = torch.device("cpu")
    
    # Initialize model
    model = TinyHybridStudent(input_dim=args.input_dim, hidden_dim=64, num_classes=2)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # Create dummy inputs matching the expected shapes
    if args.can_window_size != CAN_WINDOW_SIZE_STANDARD:
        raise ValueError(
            f"Invalid CAN window size {args.can_window_size}. Standard is {CAN_WINDOW_SIZE_STANDARD}."
        )
    # CAN: (Batch, Window, Features) -> (1, 100, 16) -- standards aligned
    dummy_can = torch.randn(1, args.can_window_size, args.input_dim, device=device)
    # ETH: (Batch, Channels, Height, Width) -> (1, 1, 32, 32) -- actual image size from dataset
    dummy_eth = torch.randn(1, 1, 32, 32, device=device)

    print(f"Exporting to ONNX format at {args.output_path}...")
    
    # Export the model
    torch.onnx.export(
        model,                                     # model being run
        (dummy_can, dummy_eth),                    # model input (or a tuple for multiple inputs)
        args.output_path,                          # where to save the model
        export_params=True,                        # store the trained parameter weights inside the model file
        opset_version=14,                          # the ONNX version to export the model to
        do_constant_folding=True,                  # whether to execute constant folding for optimization
        input_names=['can_input', 'eth_input'],    # the model's input names
        output_names=['output'],                   # the model's output names
        dynamic_axes={
            'can_input': {0: 'batch_size'},        # variable length axes
            'eth_input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("Export successful!")
    print(f"The model is ready for deployment using ONNX Runtime, TensorRT, or OpenVINO.")

if __name__ == "__main__":
    main()
