"""Distributed shape loading with an explicit, resumable consumed-batch cursor."""

import math

import torch
from torch.utils.data import DataLoader, Sampler

from .objaverse import PairedShapeDataset


class StatefulBatchSampler(Sampler):
    """Prefetch may advance the iterator; only advance() commits consumption.

    This separates worker prefetch from the checkpoint cursor. Resuming with the
    same rank count, batch size, dataset order and seed starts at the next batch.
    """
    def __init__(self, dataset_size, batch_size, world_size=1, rank=0, seed=42, fingerprint=""):
        if dataset_size <= 0 or batch_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid dataset size, batch size or distributed rank")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.fingerprint = fingerprint
        self.batches_per_epoch = math.ceil(dataset_size / (batch_size * world_size))
        self.epoch = 0
        self.batch_index = 0

    def _indices(self, epoch):
        generator = torch.Generator().manual_seed(self.seed + epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        total = self.batches_per_epoch * self.batch_size * self.world_size
        indices = (indices * math.ceil(total / len(indices)))[:total]
        local_size = self.batches_per_epoch * self.batch_size
        return indices[self.rank * local_size:(self.rank + 1) * local_size]

    def __iter__(self):
        epoch, start = self.epoch, self.batch_index
        while True:
            indices = self._indices(epoch)
            for batch_index in range(start, self.batches_per_epoch):
                offset = batch_index * self.batch_size
                yield indices[offset:offset + self.batch_size]
            epoch, start = epoch + 1, 0

    def __len__(self):
        return self.batches_per_epoch

    def advance(self):
        self.batch_index += 1
        if self.batch_index == self.batches_per_epoch:
            self.epoch += 1
            self.batch_index = 0

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "dataset_size", "batch_size", "world_size", "rank", "seed", "fingerprint", "epoch", "batch_index"
        )}

    def load_state_dict(self, state):
        for key in ("dataset_size", "batch_size", "world_size", "rank", "seed", "fingerprint"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Cannot resume with a different sampler {key}")
        if state["epoch"] < 0 or not 0 <= state["batch_index"] < self.batches_per_epoch:
            raise ValueError("Invalid sampler checkpoint cursor")
        self.epoch, self.batch_index = state["epoch"], state["batch_index"]


def load_training_data(config, batch_size, world_size=1, rank=0):
    dataset = PairedShapeDataset(config.data.root, config.data.category, config.data.category_filter)
    sampler = StatefulBatchSampler(len(dataset), batch_size, world_size, rank,
                                   config.training.seed, dataset.fingerprint)
    # DataLoader iterator/worker seed generation must not consume model RNG.
    generator = torch.Generator().manual_seed(config.training.seed + rank)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=config.data.num_workers,
                      pin_memory=True, generator=generator)
