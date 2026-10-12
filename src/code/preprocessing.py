import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


DATASET_PATH: str = os.path.join(
    os.getcwd(),
    "data",
    "raw",
    "whu_building"
)

PROCESSED_DATASET_PATH: str = os.path.join(
    os.getcwd(),
    "data",
    "processed",
    "whu_building"
)

# The raw WHU Building download only contains a "train" folder,
# so train / val / test are carved out of it.
RAW_SOURCE_SPLIT: str = "train"
SPLIT_RATIOS: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED: int = 42

TILE_SIZE: int = 512


preprocess_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image_mask(
    image_path: str,
    mask_path: str,
    tile_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    image: Image.Image = Image.open(image_path).convert("RGB")
    mask: Image.Image = Image.open(mask_path).convert("L")

    image = image.resize(
        (tile_size, tile_size),
        Image.Resampling.BILINEAR
    )

    mask = mask.resize(
        (tile_size, tile_size),
        Image.Resampling.NEAREST
    )

    # ToTensor() already returns torch.Tensor.
    image_tensor: torch.Tensor = preprocess_transforms(image)

    # Convert mask to NumPy first.
    mask_array: np.ndarray = np.asarray(mask)

    # Convert binary mask to torch.Tensor.
    mask_tensor: torch.Tensor = torch.from_numpy(
        (mask_array == 255).astype(np.float32)
    )

    return image_tensor, mask_tensor


def save_preprocessed_data(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    output_path: str
) -> None:

    torch.save(
        (image_tensor, mask_tensor),
        output_path
    )


def build_splits() -> Dict[str, List[str]]:
    """Randomly split the raw tiles into train / val / test."""

    image_dir: str = os.path.join(
        DATASET_PATH,
        RAW_SOURCE_SPLIT,
        "Image"
    )

    mask_dir: str = os.path.join(
        DATASET_PATH,
        RAW_SOURCE_SPLIT,
        "Mask"
    )

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    ])

    mask_files = set(
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    )

    missing = [f for f in image_files if f not in mask_files]

    if missing:
        print(
            f"[WARNING] {len(missing)} images have no mask "
            f"and will be skipped."
        )

    image_files = [f for f in image_files if f in mask_files]

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(image_files)

    total = len(image_files)
    n_train = round(total * SPLIT_RATIOS["train"])
    n_val = round(total * SPLIT_RATIOS["val"])

    return {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:],
    }


def preprocess_files(split: str, image_files: List[str]) -> None:

    image_dir: str = os.path.join(
        DATASET_PATH,
        RAW_SOURCE_SPLIT,
        "Image"
    )

    mask_dir: str = os.path.join(
        DATASET_PATH,
        RAW_SOURCE_SPLIT,
        "Mask"
    )

    processed_dir: str = os.path.join(
        PROCESSED_DATASET_PATH,
        split
    )

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    # Never mix new tiles with stale files from an earlier run.
    existing = [
        f for f in os.listdir(processed_dir)
        if f.endswith(".pt")
    ]

    if existing:
        raise RuntimeError(
            f"{processed_dir} already contains {len(existing)} .pt "
            f"files. Delete the folder before re-running "
            f"preprocessing."
        )

    building_fraction_sum: float = 0.0
    tiles_with_buildings: int = 0

    for image_file in image_files:

        image_path: str = os.path.join(
            image_dir,
            image_file
        )

        mask_path: str = os.path.join(
            mask_dir,
            image_file
        )

        image_tensor, mask_tensor = preprocess_image_mask(
            image_path,
            mask_path,
            TILE_SIZE
        )

        output_path: str = os.path.join(
            processed_dir,
            f"{image_file}.pt"
        )

        save_preprocessed_data(
            image_tensor,
            mask_tensor,
            output_path
        )

        building_fraction_sum += mask_tensor.mean().item()
        tiles_with_buildings += int(mask_tensor.max().item() > 0)

    total = len(image_files)
    mean_fraction = building_fraction_sum / max(total, 1)

    print(
        f"[INFO] {split}: {total} tiles | "
        f"{tiles_with_buildings} with buildings | "
        f"mean building fraction {mean_fraction:.4f}"
    )

    # Sanity check: an empty or near-empty label set is the exact
    # failure that produces "100% accuracy, empty predictions".
    if tiles_with_buildings == 0 or mean_fraction < 0.005:
        raise RuntimeError(
            f"Split '{split}' has almost no building pixels "
            f"(mean fraction {mean_fraction:.5f}). "
            f"Check the raw masks before training."
        )

    if not 0.01 <= mean_fraction <= 0.40:
        print(
            f"[WARNING] {split}: mean building fraction "
            f"{mean_fraction:.4f} is outside the usual range "
            f"(about 0.05 to 0.20). Double-check the masks."
        )


def preprocess_all_splits() -> None:

    splits = build_splits()

    for split, files in splits.items():
        preprocess_files(split, files)

    print("[INFO] Finished preprocessing all splits.")


if __name__ == "__main__":
    preprocess_all_splits()