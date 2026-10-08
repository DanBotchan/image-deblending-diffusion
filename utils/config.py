"""
Experiment configuration class for the deblend package.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import DictConfig

from . import Device, TRANSFORM
from .file import save_yaml, save_json, load_json

if TYPE_CHECKING:
    from alphas.alpha_maps import AlphaMap


class ExpConfig:
    """
    Experiment configuration class.

    Manages configuration, paths, and experiment state for deblending runs.
    Works with both Stage 1 (latent) and Stage 2 (pixel) pipelines.
    """

    def __init__(self,
                 cfg: Dict[str, Any],
                 device: Device,
                 ):
        self.device = device
        self.config = cfg
        self.paths = dict()

        self.pair_num: Optional[int] = None
        self.smp_num: Optional[int] = None

    def __contains__(self, item: str) -> bool:
        return item in self.config

    def __getitem__(self, item: str) -> Any:
        return self.config[item]

    def set_pair(self, pair_num: int) -> None:
        """Set the current pair number."""
        self.pair_num = pair_num

    def set_smp(self, smp_num: int) -> None:
        """Set the current sample number."""
        self.smp_num = smp_num

    def set_paths(self, mkdirs: bool) -> None:
        """
        Set up output paths for the experiment.

        Args:
            mkdirs: Whether to create directories
        """
        general = self['general']
        results_dir = Path(general['output_dir'])
        results_dir.mkdir(exist_ok=True, parents=True)
        self.paths['root'] = results_dir
        for img_dir in ['results', 'progress']:
            self.paths[img_dir] = results_dir / img_dir
            if mkdirs:
                self.paths[img_dir].mkdir(exist_ok=True, parents=True)

    def get_path(self,
                 key: str,
                 include_smp: bool = False) -> Path:
        """
        Get a path for storing results.

        Args:
            key: Path key ('root', 'results', or 'progress')
            include_smp: Whether to include sample number in path

        Returns:
            Path object
        """
        if key == 'root':
            return self.paths['root']
        else:
            path = self.paths[key] / f"pair_{self.pair_num:03}"
            if include_smp:
                path = path / f"smp_{self.smp_num:04}"
            return path

    def get_device(self) -> Device:
        """Get the device for this experiment."""
        return self.device

    def save_config(self) -> None:
        """Save the configuration to a YAML file."""
        save_yaml(data=self.config,
                  file_path=self.get_path('root') / 'config.yaml')


def get_config_class(cfg: Union[Dict, DictConfig],
                     device: Device) -> ExpConfig:
    """
    Create an ExpConfig instance from configuration.

    Args:
        cfg: Configuration dictionary
        device: Device to use

    Returns:
        ExpConfig instance
    """
    return ExpConfig(cfg=cfg, device=device)


def reload_from_disk(img: torch.Tensor,
                     device: Device,
                     img_path: str = None) -> torch.Tensor:
    """
    Reload the blended image from disk.

    This is used to ensure consistent image processing by saving and
    reloading through PIL, which quantizes to uint8.

    Args:
        img: Image tensor to save and reload
        device: Device to load image to
        img_path: Temporary path for saving (auto-generated if None)

    Returns:
        Reloaded image tensor
    """
    import tempfile

    norm_transform = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    pil_transform = transforms.ToPILImage()
    reverse_transform = transforms.Compose([norm_transform, pil_transform])
    img_pil = reverse_transform(img)

    if img_path is not None:
        img_path = Path(img_path)
        if not img_path.parent.exists():
            img_path.parent.mkdir(parents=True, exist_ok=True)
        img_pil.save(img_path)
        return TRANSFORM(Image.open(img_path)).to(device)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as f:
        img_pil.save(f.name)
        return TRANSFORM(Image.open(f.name)).to(device)


def save_details_json(data_item: Dict[str, Any],
                      save_to: Path,
                      sample_time: float,
                      smp_seed: int,
                      metrics: Dict[str, Any] = None) -> None:
    """
    Save experiment details to a JSON file.

    Args:
        data_item: Data item dictionary
        save_to: Directory to save to
        sample_time: Time taken for sampling
        smp_seed: Random seed used
        metrics: Optional evaluation metrics dict
    """
    # Handle both tensor and int seeds
    if isinstance(smp_seed, torch.Tensor):
        smp_seed = smp_seed.item()

    details = {
        'image1_path': data_item['image1_path'][0] if isinstance(data_item['image1_path'], list) else data_item['image1_path'],
        'image1_class': data_item['image1_class'][0] if isinstance(data_item['image1_class'], list) else data_item['image1_class'],
        'image1_prompt': data_item['image1_prompt'][0] if isinstance(data_item['image1_prompt'], list) else data_item['image1_prompt'],
        'image2_path': data_item['image2_path'][0] if isinstance(data_item['image2_path'], list) else data_item['image2_path'],
        'image2_class': data_item['image2_class'][0] if isinstance(data_item['image2_class'], list) else data_item['image2_class'],
        'image2_prompt': data_item['image2_prompt'][0] if isinstance(data_item['image2_prompt'], list) else data_item['image2_prompt'],
        'sample_time': sample_time,
        'smp_seed': smp_seed,
    }

    # Add sample paths if available (Stage 2)
    if 'sample1_path' in data_item:
        details['sample1_path'] = data_item['sample1_path'][0] if isinstance(data_item['sample1_path'], list) else data_item['sample1_path']
    if 'sample2_path' in data_item:
        details['sample2_path'] = data_item['sample2_path'][0] if isinstance(data_item['sample2_path'], list) else data_item['sample2_path']
    if 'assignment_swapped' in data_item:
        assignment_swapped = data_item['assignment_swapped']
        if isinstance(assignment_swapped, torch.Tensor):
            assignment_swapped = assignment_swapped.item()
        elif isinstance(assignment_swapped, (list, tuple)):
            assignment_swapped = assignment_swapped[0]
        details['assignment_swapped'] = bool(assignment_swapped)

    if metrics is not None:
        details['metrics'] = metrics

    save_json(save_to / 'details.json', details)


def save_alphas_params(gt: "AlphaMap",
                       pred: "AlphaMap",
                       alpha_list: list,
                       results_path: Union[Path, str],
                       init_pred_map: "AlphaMap" = None) -> None:
    """
    Save alpha map parameters and visualizations.

    Args:
        gt: Ground truth alpha map
        pred: Predicted alpha map
        alpha_list: List of alpha values over iterations
        results_path: Directory to save to
        init_pred_map: Initial predicted alpha map (Stage 2 only)
    """
    results_path = Path(results_path)

    def plot_map(alpha_map):
        fig, ax = plt.subplots(1)
        img = alpha_map.detach().cpu().numpy()
        img_plot = ax.imshow(img, vmin=0, vmax=1)
        ax.set_title(f'Min: {img.min():.3f}, Max: {img.max():.3f}')
        img_plot.set_cmap('gray')
        return fig

    def plot_line(alpha_list):
        fig, ax = plt.subplots(1)
        ax.plot(range(len(alpha_list)), alpha_list)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Alpha')
        return fig

    # Convert alpha list to means
    alpha_list = [
        alpha.detach().cpu().numpy().mean()
        if isinstance(alpha, torch.Tensor) else alpha.mean()
        for alpha in alpha_list
    ]

    def get_params(alpha_map: "AlphaMap"):
        """Get parameters with support for both scalar and spatial maps."""
        if hasattr(alpha_map, 'get_save_parameters'):
            return {
                'type': alpha_map.map_type,
                'parameters': alpha_map.get_save_parameters(),
                'raw_parameters': alpha_map.get_save_raw_parameters(),
                'init_parameters': alpha_map.get_save_init_parameters(),
            }
        else:
            n = lambda x: x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
            return {
                'type': alpha_map.map_type,
                'parameters': [n(param) for param in alpha_map.get_scaled_parameters() if param.numel() > 0],
                'raw_parameters': [n(param) for param in alpha_map.get_parameters()],
                'init_parameters': [n(param) for param in alpha_map.init_params],
            }

    alphas_dict = {}
    alphas_dict['pred'] = get_params(alpha_map=pred)
    plot_map(pred.forward()).savefig(results_path / 'pred_map.png')
    plt.close()

    plot_line(alpha_list=alpha_list).savefig(results_path / 'alpha_evolution.png')
    plt.close()

    if gt is not None:
        alphas_dict['gt'] = get_params(alpha_map=gt)
        plot_map(gt.forward()).savefig(results_path / 'gt_map.png')
        plt.close()

    if init_pred_map is not None:
        alphas_dict['init_pred_map'] = get_params(alpha_map=init_pred_map)
        plot_map(init_pred_map.forward()).savefig(results_path / 'init_pred_map.png')
        plt.close()

    save_json(file_path=results_path / 'alphas.json', data=alphas_dict)


def load_details_json(path: Path) -> Dict[str, Any]:
    """Load experiment details from JSON file."""
    return load_json(path / 'details.json')


def load_alphas_json(path: Path) -> Dict[str, Any]:
    """Load alpha parameters from JSON file."""
    return load_json(path / 'alphas.json')
