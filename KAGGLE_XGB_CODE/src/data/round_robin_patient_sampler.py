"""
Round-robin patient mini-block batch sampler for LazyEEGDataset.

The sampler:
1. shuffles patient order each epoch;
2. shuffles windows within each patient;
3. yields a limited number of batches from each patient;
4. cycles across patients until all windows are consumed.

This avoids global random access across compressed NPZ files while providing
better patient mixing than processing one complete patient at a time.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from src.data.lazy_eeg_dataset import LazyEEGDataset


class RoundRobinPatientBatchSampler(Sampler[list[int]]):
    """Yield patient-homogeneous mini-batches in shuffled round-robin blocks."""

    def __init__(
        self,
        dataset: LazyEEGDataset,
        *,
        batch_size: int,
        batches_per_patient_block: int = 8,
        shuffle_patients: bool = True,
        shuffle_within_patient: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset, LazyEEGDataset):
            raise TypeError(
                "RoundRobinPatientBatchSampler requires LazyEEGDataset."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if batches_per_patient_block < 1:
            raise ValueError(
                "batches_per_patient_block must be at least 1."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.batches_per_patient_block = int(
            batches_per_patient_block
        )
        self.shuffle_patients = bool(shuffle_patients)
        self.shuffle_within_patient = bool(shuffle_within_patient)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be zero or greater.")
        self.epoch = int(epoch)

    def _patient_batches(
        self,
        rng: np.random.Generator,
    ) -> list[deque[list[int]]]:
        all_patient_batches: list[deque[list[int]]] = []

        for info in self.dataset.patient_files:
            indices = np.arange(
                info.global_start,
                info.global_end,
                dtype=np.int64,
            )

            if self.shuffle_within_patient:
                rng.shuffle(indices)

            batches: deque[list[int]] = deque()

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                batches.append(batch.astype(int).tolist())

            all_patient_batches.append(batches)

        return all_patient_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        patient_batches = self._patient_batches(rng)

        active = [
            patient_index
            for patient_index, batches in enumerate(patient_batches)
            if batches
        ]

        if self.shuffle_patients:
            rng.shuffle(active)

        while active:
            next_active: list[int] = []

            for patient_index in active:
                batches = patient_batches[patient_index]

                for _ in range(self.batches_per_patient_block):
                    if not batches:
                        break
                    yield batches.popleft()

                if batches:
                    next_active.append(patient_index)

            active = next_active

            if self.shuffle_patients and len(active) > 1:
                rng.shuffle(active)

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
