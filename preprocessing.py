from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import cv2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLAHETransform:
    """Apply CLAHE to reduce sensitivity to lighting swings."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("CLAHETransform expects a PIL.Image")

        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if img_np.ndim == 2:  # grayscale
            img_np = clahe.apply(img_np)
        else:  # apply per channel
            channels = [clahe.apply(channel) for channel in cv2.split(img_np)]
            img_np = cv2.merge(channels)

        return Image.fromarray(img_np)


def load_mask_tensor(mask_path: str, target_size: tuple[int, int] = (256, 256), num_channels: int = 3) -> torch.Tensor:
    mask_pil = Image.open(mask_path).convert("L").resize(target_size)
    mask_array = np.array(mask_pil, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array)
    # Expand mask to match channel count (C, H, W)
    mask_tensor = mask_tensor.unsqueeze(0).repeat(num_channels, 1, 1)
    return mask_tensor


class ApplyMaskTensor:
    """Multiply an image tensor by a static mask."""

    def __init__(self, mask_tensor: torch.Tensor):
        self.mask_tensor = mask_tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.mask_tensor


def build_transforms(mask_path: str, train: bool = True) -> T.Compose:
    """Create a transform pipeline shared by training and inference."""

    mask_tensor = load_mask_tensor(mask_path)

    base_transforms = [
        T.Grayscale(num_output_channels=3),
        T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        CLAHETransform(),
    ]

    if train:
        base_transforms += [
            T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35)], p=0.65),
            T.RandomAutocontrast(p=0.35),
            T.RandomAdjustSharpness(1.5, p=0.2),
            T.RandomRotation(3, interpolation=InterpolationMode.BILINEAR, fill=0),
            T.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            T.RandomHorizontalFlip(p=0.5),
        ]

    base_transforms += [
        T.ToTensor(),
        ApplyMaskTensor(mask_tensor),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return T.Compose(base_transforms)
