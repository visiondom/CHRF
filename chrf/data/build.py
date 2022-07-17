import itertools

import numpy as np
import torch.utils.data
from benchmark.utils.comm import get_world_size
from benchmark.utils.env import seed_all_rng

from .catalog import DatasetCatalog, MapperCatalog
from .common import DatasetFromList, MapDataset
from .samplers import InferenceSampler, TrainingSampler


def get_dataset_dicts(dataset_names):
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    return dataset_dicts


def build_batch_data_loader(dataset, sampler, total_batch_size, num_workers=0):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )


def build_train_loader(cfg, mapper=None):
    dataset_dicts = get_dataset_dicts(cfg.DATASETS.TRAIN)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler = TrainingSampler(len(dataset))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_dataset_dicts(cfg.DATASETS.TEST)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = MapperCatalog.get(dataset_name)(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.TEST.IMS_PER_PROC, drop_last=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
