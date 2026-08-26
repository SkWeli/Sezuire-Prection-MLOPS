"""
EEG feature extraction for hierarchical XGBoost experiments.

Input window:
    (20, 512)

Sampling rate:
    128 Hz

Per-channel features:
    1. Relative delta power
    2. Relative theta power
    3. Relative alpha power
    4. Relative beta power
    5. Relative gamma power
    6. Spectral entropy
    7. Hjorth mobility
    8. Hjorth complexity
    9. Mean line length
    10. Skewness
    11. Kurtosis

20 channels x 11 features = 220 features/window.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


EPS = 1e-12

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

TOTAL_POWER_RANGE = (0.5, 45.0)


def zscore_channel(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score normalization."""
    x = np.asarray(x, dtype=np.float64)

    mean = np.mean(x)
    std = np.std(x)

    if std < EPS:
        return np.zeros_like(x, dtype=np.float64)

    return (x - mean) / std


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Numerically integrate PSD values."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))


def calculate_psd(
    x: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Welch PSD once for reuse by spectral features."""
    nperseg = min(256, len(x))

    freqs, psd = welch(
        x,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )

    return freqs, psd


def relative_bandpowers_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
) -> dict[str, float]:
    """Calculate relative band power from an existing PSD."""
    total_mask = (
        (freqs >= TOTAL_POWER_RANGE[0])
        & (freqs <= TOTAL_POWER_RANGE[1])
    )

    total_power = _integrate(
        psd[total_mask],
        freqs[total_mask],
    )

    total_power = max(total_power, EPS)

    results: dict[str, float] = {}

    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)

        if np.sum(mask) < 2:
            band_power = 0.0
        else:
            band_power = _integrate(
                psd[mask],
                freqs[mask],
            )

        results[band_name] = band_power / total_power

    return results


def spectral_entropy_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
) -> float:
    """Normalized spectral entropy between 0.5 and 45 Hz."""
    mask = (
        (freqs >= TOTAL_POWER_RANGE[0])
        & (freqs <= TOTAL_POWER_RANGE[1])
    )

    p = psd[mask].astype(np.float64)

    power_sum = np.sum(p)

    if power_sum <= EPS or len(p) <= 1:
        return 0.0

    p = p / power_sum
    p = np.clip(p, EPS, None)

    entropy = -np.sum(p * np.log(p))
    entropy /= np.log(len(p))

    return float(entropy)


def hjorth_parameters(
    x: np.ndarray,
) -> tuple[float, float]:
    """Calculate Hjorth mobility and complexity."""
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    if var_x <= EPS:
        return 0.0, 0.0

    mobility = np.sqrt(
        var_dx / (var_x + EPS)
    )

    if var_dx <= EPS or mobility <= EPS:
        complexity = 0.0
    else:
        mobility_dx = np.sqrt(
            var_ddx / (var_dx + EPS)
        )

        complexity = mobility_dx / (mobility + EPS)

    return float(mobility), float(complexity)


def mean_line_length(x: np.ndarray) -> float:
    """Mean absolute change between consecutive samples."""
    if len(x) < 2:
        return 0.0

    return float(
        np.mean(np.abs(np.diff(x)))
    )


def extract_channel_features(
    channel: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Extract 11 features from one EEG channel."""
    x = zscore_channel(channel)

    # Welch PSD is calculated only once.
    freqs, psd = calculate_psd(
        x,
        sfreq,
    )

    bandpowers = relative_bandpowers_from_psd(
        freqs,
        psd,
    )

    entropy = spectral_entropy_from_psd(
        freqs,
        psd,
    )

    mobility, complexity = hjorth_parameters(x)

    line_length = mean_line_length(x)

    channel_skew = float(
        skew(x, bias=False)
    )

    channel_kurtosis = float(
        kurtosis(
            x,
            fisher=True,
            bias=False,
        )
    )

    features = np.array(
        [
            bandpowers["delta"],
            bandpowers["theta"],
            bandpowers["alpha"],
            bandpowers["beta"],
            bandpowers["gamma"],
            entropy,
            mobility,
            complexity,
            line_length,
            channel_skew,
            channel_kurtosis,
        ],
        dtype=np.float32,
    )

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def extract_window_features(
    window: np.ndarray,
    sfreq: float = 128.0,
) -> np.ndarray:
    """
    Convert one EEG window into 220 features.

    Expected input:
        (20, 512)

    Output:
        (220,)
    """
    window = np.asarray(window)

    if window.ndim != 2:
        raise ValueError(
            f"Expected 2D EEG window, got shape {window.shape}"
        )

    n_channels, n_samples = window.shape

    if n_channels != 20:
        raise ValueError(
            f"Expected 20 EEG channels, got {n_channels}"
        )

    if n_samples != 512:
        raise ValueError(
            f"Expected 512 samples, got {n_samples}"
        )

    all_features = []

    for channel_index in range(n_channels):
        features = extract_channel_features(
            window[channel_index],
            sfreq,
        )

        all_features.append(features)

    feature_vector = np.concatenate(
        all_features,
        axis=0,
    )

    if feature_vector.shape != (220,):
        raise RuntimeError(
            f"Unexpected feature shape: {feature_vector.shape}"
        )

    return feature_vector.astype(
        np.float32,
        copy=False,
    )


def build_feature_names(
    ch_names: list[str] | np.ndarray,
) -> list[str]:
    """Generate deterministic names for all 220 features."""
    channel_feature_names = [
        "rel_delta_power",
        "rel_theta_power",
        "rel_alpha_power",
        "rel_beta_power",
        "rel_gamma_power",
        "spectral_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "mean_line_length",
        "skewness",
        "kurtosis",
    ]

    names: list[str] = []

    for channel in ch_names:
        channel = str(channel)

        for feature_name in channel_feature_names:
            names.append(
                f"{channel}__{feature_name}"
            )

    return names