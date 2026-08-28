from typing import Callable

from core.registry import Registry
from utils.exp_ops import ExpConfig

CATEGORY = 'dataset'


def register_dataset_type(name: str):
    return Registry.register(CATEGORY, name)


def get_dataset(exp_config: ExpConfig,
                transform: Callable,
                ):
    dataset_config = exp_config['datasets']
    dataset_class = Registry.get(CATEGORY, dataset_config['dataset_type'])

    return dataset_class(dataset_path=dataset_config['dataset_path'],
                         annotations_path=dataset_config['annotations_path'],
                         img_size=dataset_config['img_size'],
                         transform=transform,
                         )
