"""
Deployment utilities for the EEG seizure research prototype.

This package contains functionality for:

- exporting the frozen PyTorch TCN model to ONNX
- validating ONNX model structure
- comparing PyTorch and ONNX Runtime outputs
- preparing later INT8 quantization experiments

Scientific status:
    This project is a research prototype.
    It is not clinically validated and is not a medical device.
"""