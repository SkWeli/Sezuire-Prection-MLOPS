#!/usr/bin/env python3
"""
FP32 ONNX Runtime offline demonstration entry point.

The shared implementation is located in:

    offline_demo.onnx_runtime_demo
"""

from __future__ import annotations

from offline_demo.config import (
    EXPECTED_FP32_ONNX_SHA256,
    FP32_ONNX_MODEL_PATH,
)

from offline_demo.onnx_runtime_demo import (
    ONNXDemoProfile,
    run_demo,
)


FP32_PROFILE = ONNXDemoProfile(
    display_name="FP32 ONNX model",
    short_name="fp32_onnx",
    model_path=FP32_ONNX_MODEL_PATH,
    expected_sha256=EXPECTED_FP32_ONNX_SHA256,
    require_quantization_nodes=False,
)


def main() -> int:
    """
    Run the verified FP32 ONNX demonstration.
    """
    return run_demo(
        FP32_PROFILE
    )


if __name__ == "__main__":
    raise SystemExit(main())