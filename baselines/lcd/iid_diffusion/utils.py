"""
Utility functions for IID image decomposition
"""
import torch
from torch.utils import data
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np


class PairedImageDataset(data.Dataset):
    """
    Dataset that loads pairs of images for training IID decomposition.

    During training, images are paired and one is designated as 'original'
    (higher pixel intensity) and one as 'added' (lower pixel intensity).
    """

    def __init__(self, folder, image_size, augment=True, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.augment = augment
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        if len(self.paths) < 2:
            raise ValueError(f"Need at least 2 images in {folder}, found {len(self.paths)}")

        print(f"Found {len(self.paths)} images in {folder}")

        # Transform for augmentation
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """
        Returns a pair of images.

        Returns:
            dict with keys:
                'x_orig': Original image (higher intensity)
                'x_add': Added image (lower intensity)
        """
        # Load first image
        path1 = self.paths[index]
        img1 = Image.open(path1).convert('RGB')
        img1 = self.transform(img1)

        # Randomly select second image (different from first)
        index2 = np.random.randint(0, len(self.paths))
        while index2 == index and len(self.paths) > 1:
            index2 = np.random.randint(0, len(self.paths))

        path2 = self.paths[index2]
        img2 = Image.open(path2).convert('RGB')
        img2 = self.transform(img2)

        # Determine which is 'original' and 'added' based on pixel intensity
        # (as mentioned in Section III.A of the paper)
        intensity1 = img1.mean().item()
        intensity2 = img2.mean().item()

        if intensity1 > intensity2:
            x_orig = img1
            x_add = img2
        else:
            x_orig = img2
            x_add = img1

        return {
            'x_orig': x_orig,
            'x_add': x_add,
        }


class BIDPairedDataset(data.Dataset):
    """
    Dataset that loads pairs of images from two separate folders for BID training.

    trainA contains original images (x_orig)
    trainB contains added images (x_add)

    During each iteration, randomly samples one image from each folder.
    """

    def __init__(self, folder, image_size, augment=True, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = Path(folder)
        self.image_size = image_size
        self.augment = augment

        # Load paths from trainA and trainB
        trainA_folder = self.folder / 'trainA'
        trainB_folder = self.folder / 'trainB'

        if not trainA_folder.exists():
            raise ValueError(f"trainA folder not found at {trainA_folder}")
        if not trainB_folder.exists():
            raise ValueError(f"trainB folder not found at {trainB_folder}")

        self.paths_orig = [p for ext in exts for p in trainA_folder.glob(f'*.{ext}')]
        self.paths_add = [p for ext in exts for p in trainB_folder.glob(f'*.{ext}')]

        if len(self.paths_orig) == 0:
            raise ValueError(f"No images found in {trainA_folder}")
        if len(self.paths_add) == 0:
            raise ValueError(f"No images found in {trainB_folder}")

        print(f"Found {len(self.paths_orig)} images in trainA (x_orig)")
        print(f"Found {len(self.paths_add)} images in trainB (x_add)")

        # Transform for augmentation
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

    def __len__(self):
        # Use the larger of the two datasets
        return max(len(self.paths_orig), len(self.paths_add))

    def __getitem__(self, index):
        """
        Returns a pair of images by randomly sampling from trainA and trainB.

        Returns:
            dict with keys:
                'x_orig': Original image from trainA
                'x_add': Added image from trainB
        """
        # Randomly sample from trainA (x_orig)
        idx_orig = np.random.randint(0, len(self.paths_orig))
        path_orig = self.paths_orig[idx_orig]
        img_orig = Image.open(path_orig).convert('RGB')
        img_orig = self.transform(img_orig)

        # Randomly sample from trainB (x_add)
        idx_add = np.random.randint(0, len(self.paths_add))
        path_add = self.paths_add[idx_add]
        img_add = Image.open(path_add).convert('RGB')
        img_add = self.transform(img_add)

        return {
            'x_orig': img_orig,
            'x_add': img_add,
        }


class TestDataset(data.Dataset):
    """
    Dataset for testing - loads single images that will be paired at test time.
    """

    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f"Found {len(self.paths)} test images in {folder}")

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img), str(path)


def cycle(dl):
    """Infinite dataloader cycle."""
    while True:
        for data in dl:
            yield data


def compute_ssim(img1, img2):
    """
    Compute SSIM between two images.

    Args:
        img1, img2: Tensors of shape [B, C, H, W] in range [-1, 1]

    Returns:
        SSIM value
    """
    try:
        from pytorch_msssim import ssim
        # Convert from [-1, 1] to [0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return ssim(img1, img2, data_range=1.0, size_average=True).item()
    except ImportError:
        print("Warning: pytorch_msssim not installed. Install with: pip install pytorch-msssim")
        return 0.0


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images.

    Args:
        img1, img2: Tensors of shape [B, C, H, W] in range [-1, 1]

    Returns:
        PSNR value in dB
    """
    # Convert from [-1, 1] to [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_lpips(img1, img2, lpips_fn):
    """
    Compute LPIPS between two images.

    Args:
        img1, img2: Tensors of shape [B, C, H, W] in range [-1, 1]
        lpips_fn: LPIPS model

    Returns:
        LPIPS value
    """
    with torch.no_grad():
        return lpips_fn(img1, img2).mean().item()


def save_image_grid(images, filepath, nrow=8):
    """
    Save a grid of images.

    Args:
        images: Tensor of shape [B, C, H, W] in range [-1, 1]
        filepath: Where to save
        nrow: Number of images per row
    """
    from torchvision.utils import save_image

    # Convert from [-1, 1] to [0, 1]
    images = (images + 1) / 2

    save_image(images, filepath, nrow=nrow)


def num_to_groups(num, divisor):
    """Split num into groups of size divisor."""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
