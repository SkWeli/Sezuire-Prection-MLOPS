"""
Memory-conscious lazy EEG dataset for patient-level NPZ files.

The current eager training pipeline loads every patient's EEG epochs,
concatenates them, and converts the complete cohort into PyTorch tensors.
That can require several copies of the full dataset in RAM.

This dataset instead:

1. Builds a lightweight global index from patient label arrays.
2. Opens a patient NPZ file only when one of its windows is requested.
3. Keeps only a small number of patient arrays in an LRU cache.
4. Applies the existing per-window, per-channel z-score normalization.
5. Returns one EEG window and label at a time.

Expected NPZ contract:
    epochs: (n_windows, 20, 512)
    labels: (n_windows,)
    sfreq:  128 Hz, when present

Class contract:
    0 = Interictal
    1 = Pre-Ictal
    2 = Ictal
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatientFileInfo:
    """Small metadata record that does not contain EEG arrays."""

    patient_id: str
    npz_path: Path
    n_windows: int
    global_start: int
    global_end: int


class LazyEEGDataset(Dataset):
    """
    Lazy dataset over multiple patient NPZ files.

    Parameters
    ----------
    npz_files:
        Patient NPZ paths included in this dataset split.

    cache_size:
        Maximum number of complete patient arrays kept in RAM.
        Begin with 1 to minimize memory usage.

    normalize:
        Apply the same normalization currently used in train.py:
        per-window, per-channel z-score over the time dimension.

    expected_channels:
        Required EEG channel count.

    expected_timepoints:
        Required samples per window.

    expected_sfreq:
        Required sampling rate when ``sfreq`` exists in the NPZ file.

    return_patient_id:
        When true, return ``(window, label, patient_id)``.
        Training should normally leave this false.
    """

    def __init__(
        self,
        npz_files: Sequence[str | Path],
        *,
        cache_size: int = 1,
        normalize: bool = True,
        expected_channels: int = 20,
        expected_timepoints: int = 512,
        expected_sfreq: float = 128.0,
        return_patient_id: bool = False,
    ) -> None:
        super().__init__()

        if cache_size < 1:
            raise ValueError("cache_size must be at least 1.")

        paths = [Path(path).resolve() for path in npz_files]

        if not paths:
            raise ValueError("At least one patient NPZ file is required.")

        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The following patient NPZ files do not exist:\n"
                + "\n".join(missing)
            )

        self.cache_size = int(cache_size)
        self.normalize = bool(normalize)
        self.expected_channels = int(expected_channels)
        self.expected_timepoints = int(expected_timepoints)
        self.expected_sfreq = float(expected_sfreq)
        self.return_patient_id = bool(return_patient_id)

        # Each worker/process receives its own cache.
        self._patient_cache: OrderedDict[
            Path,
            tuple[np.ndarray, np.ndarray],
        ] = OrderedDict()

        self.patient_files: list[PatientFileInfo] = []
        self._global_ends: list[int] = []
        self._labels_by_patient: list[np.ndarray] = []

        global_start = 0

        for npz_path in paths:
            patient_id = npz_path.stem

            # Reading labels does not materialize the much larger epochs array.
            with np.load(npz_path, allow_pickle=False) as data:
                self._validate_required_keys(npz_path, data.files)

                labels = np.asarray(data["labels"], dtype=np.int64)

                if labels.ndim != 1:
                    raise ValueError(
                        f"{npz_path}: labels must be one-dimensional, "
                        f"found shape {labels.shape}."
                    )

                if labels.size == 0:
                    raise ValueError(
                        f"{npz_path}: patient contains no labelled windows."
                    )

                invalid_labels = np.setdiff1d(
                    np.unique(labels),
                    np.array([0, 1, 2]),
                )

                if invalid_labels.size:
                    raise ValueError(
                        f"{npz_path}: invalid labels "
                        f"{invalid_labels.tolist()}; expected only 0, 1, 2."
                    )

                if "sfreq" in data.files:
                    sfreq = float(np.asarray(data["sfreq"]).reshape(-1)[0])

                    if not np.isclose(
                        sfreq,
                        self.expected_sfreq,
                        rtol=0.0,
                        atol=1e-6,
                    ):
                        raise ValueError(
                            f"{npz_path}: expected sfreq "
                            f"{self.expected_sfreq}, found {sfreq}."
                        )

            n_windows = int(labels.shape[0])
            global_end = global_start + n_windows

            self.patient_files.append(
                PatientFileInfo(
                    patient_id=patient_id,
                    npz_path=npz_path,
                    n_windows=n_windows,
                    global_start=global_start,
                    global_end=global_end,
                )
            )

            self._global_ends.append(global_end)
            self._labels_by_patient.append(labels)
            global_start = global_end

        self._length = global_start

        # This is only one int64 label per window, not the EEG arrays.
        self._all_labels = np.concatenate(self._labels_by_patient).astype(
            np.int64,
            copy=False,
        )

    @staticmethod
    def _validate_required_keys(
        npz_path: Path,
        available_keys: Iterable[str],
    ) -> None:
        keys = set(available_keys)
        required = {"epochs", "labels"}
        missing = required - keys

        if missing:
            raise KeyError(
                f"{npz_path}: missing required NPZ keys "
                f"{sorted(missing)}."
            )

    def __len__(self) -> int:
        return self._length

    def get_all_labels(self, *, copy: bool = True) -> np.ndarray:
        """
        Return labels without loading any EEG epochs.

        This is useful for class counts, class weights and sampling weights.
        """

        if copy:
            return self._all_labels.copy()

        return self._all_labels

    def get_patient_ids(self) -> list[str]:
        """Return patient IDs in dataset order."""

        return [info.patient_id for info in self.patient_files]

    def get_patient_class_counts(self) -> dict[str, list[int]]:
        """Return [interictal, pre-ictal, ictal] counts for each patient."""

        result: dict[str, list[int]] = {}

        for info, labels in zip(
            self.patient_files,
            self._labels_by_patient,
        ):
            counts = np.bincount(labels, minlength=3)
            result[info.patient_id] = counts.astype(int).tolist()

        return result

    def clear_cache(self) -> None:
        """Remove all currently cached patient arrays."""

        self._patient_cache.clear()

    @property
    def cached_patient_ids(self) -> list[str]:
        """Patient IDs currently resident in this process's RAM cache."""

        return [path.stem for path in self._patient_cache.keys()]

    def _resolve_global_index(
        self,
        index: int,
    ) -> tuple[int, int]:
        """
        Convert a global dataset index into:

        ``(patient_position, local_window_index)``.
        """

        if index < 0:
            index += self._length

        if index < 0 or index >= self._length:
            raise IndexError(
                f"Dataset index {index} is outside [0, {self._length})."
            )

        patient_position = int(
            np.searchsorted(
                self._global_ends,
                index,
                side="right",
            )
        )

        info = self.patient_files[patient_position]
        local_index = index - info.global_start

        return patient_position, local_index

    def _load_patient(
        self,
        patient_position: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load one patient or return it from the LRU cache.

        Because compressed NPZ arrays cannot be truly memory-mapped,
        the complete epochs array for one patient is decompressed.
        Only ``cache_size`` patients are retained at one time.
        """

        info = self.patient_files[patient_position]
        npz_path = info.npz_path

        if npz_path in self._patient_cache:
            epochs, labels = self._patient_cache.pop(npz_path)

            # Reinsert to mark this entry as most recently used.
            self._patient_cache[npz_path] = (epochs, labels)
            return epochs, labels

        with np.load(npz_path, allow_pickle=False) as data:
            epochs = np.asarray(data["epochs"])
            labels = np.asarray(data["labels"], dtype=np.int64)

        expected_shape = (
            info.n_windows,
            self.expected_channels,
            self.expected_timepoints,
        )

        if epochs.shape != expected_shape:
            raise ValueError(
                f"{npz_path}: expected epochs shape {expected_shape}, "
                f"found {epochs.shape}."
            )

        if labels.shape != (info.n_windows,):
            raise ValueError(
                f"{npz_path}: expected labels shape "
                f"{(info.n_windows,)}, found {labels.shape}."
            )

        self._patient_cache[npz_path] = (epochs, labels)

        while len(self._patient_cache) > self.cache_size:
            self._patient_cache.popitem(last=False)

        return epochs, labels

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        patient_position, local_index = self._resolve_global_index(index)

        epochs, labels = self._load_patient(patient_position)
        info = self.patient_files[patient_position]

        # Convert only one window into float32.
        window_np = np.asarray(
            epochs[local_index],
            dtype=np.float32,
        )

        if not np.isfinite(window_np).all():
            raise ValueError(
                f"{info.npz_path}: non-finite values found at "
                f"local window {local_index}."
            )

        window = torch.from_numpy(window_np.copy())

        if self.normalize:
            # Match the existing train.py implementation:
            # normalize independently for each channel over time.
            mean = window.mean(dim=-1, keepdim=True)
            std = window.std(dim=-1, keepdim=True)
            window = (window - mean) / (std + 1e-6)

        # Model input contract: (1, 20, 512)
        window = window.unsqueeze(0)

        label = torch.tensor(
            int(labels[local_index]),
            dtype=torch.long,
        )

        if self.return_patient_id:
            return window, label, info.patient_id

        return window, label