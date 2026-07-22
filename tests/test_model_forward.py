"""
Lightweight architecture checks for the EEG classification models.

These tests are designed for GitHub Actions and local development.

They do not require:

- processed EEG datasets;
- trained checkpoints;
- CUDA;
- MLflow;
- Kaggle files.

They verify only that each architecture accepts the expected EEG input
shape and produces finite three-class logits.
"""

from __future__ import annotations

import pytest
import torch

from src.models.cnn import SeizureCNN
from src.models.tcn import SeizureTCN


# ---------------------------------------------------------------------------
# Shared model contract
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

    A fixed seed ensures repeated local and CI runs receive the same tensor.
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
    Run common forward-pass assertions for an EEG model.
    """
    model.eval()

    with torch.inference_mode():
        logits = model(
            input_tensor
        )

    assert tuple(logits.shape) == EXPECTED_OUTPUT_SHAPE

    assert torch.isfinite(logits).all(), (
        "Model output contains NaN or infinity."
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count > 0, (
        "Model has no trainable parameters."
    )


def test_cnn_forward_pass() -> None:
    """
    Verify the CNN accepts [batch, 1, 20, 512] input and emits
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
    Verify the TCN accepts [batch, 1, 20, 512] input and emits
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
    Confirm that an invalid input channel contract does not silently
    produce an apparently valid three-class result.

    The architecture may raise RuntimeError or ValueError depending on
    the layer at which the shape mismatch is detected.
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