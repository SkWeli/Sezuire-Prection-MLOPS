#!/usr/bin/env python3
"""
Static INT8 QDQ ONNX Runtime offline demonstration entry point.

The shared implementation is located in:

    offline_demo.onnx_runtime_demo
"""

from __future__ import annotations

from offline_demo.config import (
    EXPECTED_INT8_ONNX_SHA256,
    INT8_ONNX_MODEL_PATH,
)

from offline_demo.onnx_runtime_demo import (
    ONNXDemoProfile,
    run_demo,
)


INT8_PROFILE = ONNXDemoProfile(
    display_name="INT8 QDQ ONNX model",
    short_name="int8_qdq_onnx",
    model_path=INT8_ONNX_MODEL_PATH,
    expected_sha256=EXPECTED_INT8_ONNX_SHA256,
    require_quantization_nodes=True,
)


def main() -> int:
    """
    Run the verified static INT8 QDQ demonstration.
    """
    return run_demo(
        INT8_PROFILE
    )


if __name__ == "__main__":
    raise SystemExit(main())