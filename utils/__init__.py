"""
Shared utilities for the deblend package.
"""
from typing import (NewType,
                    Union,
                    Literal,
                    )

import torch
import torchvision.transforms as transforms

from .common import (flush,
                     set_seed,
                     )
from .image import (show_tensor,
                    save_results,
                    clear_color,
                    img_to_save,
                    normalize_np,
                    )
from .logger import get_logger
from .file import (save_yaml,
                   load_yaml,
                   save_json,
                   load_json,
                   save_pkl,
                   load_pkl,
                   )


TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))
                                ])

Device = NewType('Device', Union[Literal['cuda', 'cpu'], str, torch.device])
