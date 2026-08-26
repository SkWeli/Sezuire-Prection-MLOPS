"""
Patient-block batch sampler for LazyEEGDataset.

Why this exists
---------------
A normal DataLoader with shuffle=True randomly mixes windows from every
patient. With compressed NPZ files and a one-patient cache, this can force
the dataset to repeatedly decompress different patient files.

This sampler instead:

1. Shuffles the order of patients each epoch.
2. Shuffles window indices inside each patient.
3. Produces batches from one patient at a time.
4. Avoids repeated patient-file decompression.
5. Preserves useful randomness for training.

This sampler is intended for LazyEEGDataset only.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import numpy as np
from torch.utils.data import Sampler

from src.data.lazy_eeg_dataset import LazyEEGDataset


class PatientBlockBatchSampler(Sampler[list[int]], Sized):
    """
    Yield batches containing windows from one patient at a time.

    Parameters
    ----------
    dataset:
        LazyEEGDataset instance.

    batch_size:
        Number of samples per batch.

    shuffle_patients:
        Shuffle patient order at the beginning of each epoch.

    shuffle_within_patient:
        Shuffle window indices independently within each patient.

    drop_last:
        Drop incomplete final batch for each patient.

    seed:
        Base random seed. The actual order changes with ``set_epoch()``.
    """

    def __init__(
        self,
        dataset: LazyEEGDataset,
        *,
        batch_size: int,
        shuffle_patients: bool = True,
        shuffle_within_patient: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset, LazyEEGDataset):
            raise TypeError(
                "PatientBlockBatchSampler requires LazyEEGDataset."
            )

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_patients = bool(shuffle_patients)
        self.shuffle_within_patient = bool(shuffle_within_patient)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Change the deterministic shuffle order for a new epoch.
        """

        if epoch < 0:
            raise ValueError("epoch must be zero or greater.")

        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        patient_positions = np.arange(
            len(self.dataset.patient_files),
            dtype=np.int64,
        )

        if self.shuffle_patients:
            rng.shuffle(patient_positions)

        for patient_position in patient_positions:
            info = self.dataset.patient_files[int(patient_position)]

            patient_indices = np.arange(
                info.global_start,
                info.global_end,
                dtype=np.int64,
            )

            if self.shuffle_within_patient:
                rng.shuffle(patient_indices)

            for batch_start in range(
                0,
                len(patient_indices),
                self.batch_size,
            ):
                batch = patient_indices[
                    batch_start : batch_start + self.batch_size
                ]

                if (
                    self.drop_last
                    and len(batch) < self.batch_size
                ):
                    continue

                yield batch.astype(int).tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                info.n_windows // self.batch_size
                for info in self.dataset.patient_files
            )

        return sum(
            math.ceil(info.n_windows / self.batch_size)
            for info in self.dataset.patient_files
        )