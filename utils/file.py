"""
File I/O utility functions for the deblend package.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Union

import yaml
from omegaconf import OmegaConf, DictConfig


def save_json(file_path: Union[str, Path],
              data: Dict[str, Any]) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as fp:
        json.dump(data, fp, indent=4)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as fp:
        return json.load(fp)


def save_yaml(data: Union[Dict[str, Any], DictConfig],
              file_path: Union[str, Path]) -> None:
    """Save data to a YAML file."""
    if isinstance(data, DictConfig):
        OmegaConf.save(data, file_path)
    else:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a YAML file."""
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_pkl(data: Any,
             path: Union[str, Path]) -> None:
    """Save data to a pickle file."""
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)


def load_pkl(path: Union[str, Path]) -> Any:
    """Load data from a pickle file."""
    with open(path, 'rb') as fp:
        return pickle.load(fp)
