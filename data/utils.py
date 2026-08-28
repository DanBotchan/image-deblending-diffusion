"""
Data loading utilities for the deblend package.
"""
from __future__ import annotations
from typing import (Callable, Optional, List, TYPE_CHECKING)

import numpy as np
import torch
from torch.utils.data import (Sampler,
                              DataLoader,
                              )

if TYPE_CHECKING:
    from utils.config import ExpConfig

from .registry import get_dataset
from .datasets import BaseDataset

class SubsetSampler(Sampler):
    def __init__(self,
                 dataset: BaseDataset,
                 subset: Optional[List[int]],
                 N_images: int,
                 shuffle: bool = False):
        self.shuffle = shuffle

        if N_images is not None and N_images < len(dataset):
            subset_indices = np.random.choice(len(dataset),
                                              size=N_images,
                                              replace=False)
        elif subset is not None:
            subset_indices = np.arange(*subset)

        else:
            subset_indices = np.arange(len(dataset))

        self.indices = subset_indices

    def __iter__(self,
                 ):
        if self.shuffle:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self,
                ) -> int:
        return len(self.indices)


def get_data_as_dataloader(exp_config: ExpConfig,
                           transform: Callable,
                           ) -> DataLoader:
    gen_conf = exp_config['general']
    test_dataset = get_dataset(exp_config=exp_config,
                               transform=transform)

    subset = gen_conf['dataset_subset'] if 'dataset_subset' in gen_conf else None
    N_images = gen_conf['N_pairs'] if subset is None else None
    subset_sampler = SubsetSampler(dataset=test_dataset,
                                   subset=subset,
                                   N_images=N_images,
                                   shuffle=gen_conf['shuffle_data'])

    dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False,
                            sampler=subset_sampler)
    return dataloader
