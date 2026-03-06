import os

from src_replica.test_onnx_replica import test_onnx_model


def test_onnx_contract_and_inference_runs():
    onnx_path = "models/student_tiny_improved.onnx"
    if not os.path.exists(onnx_path):
        return
    test_onnx_model(onnx_path=onnx_path)
