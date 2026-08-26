"""
Lightweight forward-pass tests for the EEG model architectures.

These tests verify the model input/output contract without requiring:

- processed EEG datasets
- trained checkpoints
- CUDA
- MLflow
- Kaggle artifacts

They are suitable for both local pytest execution and GitHub Actions.
"""

from __future__ import annotations

import pytest
import torch

from src.models.cnn import SeizureCNN
from src.models.tcn import SeizureTCN


# ---------------------------------------------------------------------------
# Shared EEG model contract
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
N_CHANNELS = 20
N_TIMEPOINTS = 512
N_CLASSES = 3

EXPECTED_INPUT_SHAPE = (
    BATCH_SIZE,
    1,
    N_CHANNELS,
    N_TIMEPOINTS,
)

EXPECTED_OUTPUT_SHAPE = (
    BATCH_SIZE,
    N_CLASSES,
)


def create_test_input() -> torch.Tensor:
    """
    Create deterministic synthetic EEG input.

    A fixed random seed ensures local and CI runs receive the same
    input tensor.
    """
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(42)

    return torch.randn(
        EXPECTED_INPUT_SHAPE,
        generator=generator,
        dtype=torch.float32,
    )


def assert_valid_model_output(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> None:
    """
    Run shared assertions for an EEG classification model.
    """
    model.eval()

    with torch.inference_mode():
        logits = model(
            input_tensor
        )

    assert tuple(logits.shape) == EXPECTED_OUTPUT_SHAPE, (
        f"Expected output shape {EXPECTED_OUTPUT_SHAPE}, "
        f"received {tuple(logits.shape)}."
    )

    assert torch.isfinite(logits).all(), (
        "Model output contains NaN or infinity."
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_parameter_count > 0, (
        "Model contains no trainable parameters."
    )


def test_cnn_forward_pass() -> None:
    """
    Verify that the CNN accepts the expected EEG tensor and produces
    three logits per sample.
    """
    model = SeizureCNN(
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        n_classes=N_CLASSES,
    )

    test_input = create_test_input()

    assert_valid_model_output(
        model,
        test_input,
    )


def test_tcn_forward_pass() -> None:
    """
    Verify that the TCN accepts the expected EEG tensor and produces
    three logits per sample.
    """
    model = SeizureTCN(
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        n_classes=N_CLASSES,
    )

    test_input = create_test_input()

    assert_valid_model_output(
        model,
        test_input,
    )


@pytest.mark.parametrize(
    "model_class",
    [
        SeizureCNN,
        SeizureTCN,
    ],
)
def test_models_reject_wrong_channel_count(
    model_class: type[torch.nn.Module],
) -> None:
    """
    Verify that an invalid 19-channel input does not silently produce
    a valid-looking prediction.
    """
    model = model_class(
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        n_classes=N_CLASSES,
    )

    invalid_input = torch.randn(
        BATCH_SIZE,
        1,
        19,
        N_TIMEPOINTS,
        dtype=torch.float32,
    )

    model.eval()

    with pytest.raises(
        (RuntimeError, ValueError)
    ):
        with torch.inference_mode():
            model(invalid_input)