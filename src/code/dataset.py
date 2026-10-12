import os
from typing import Tuple

import torch
from torch.utils.data import Dataset


class WHUBuildingDataset(Dataset):

    def __init__(
        self,
        split: str = "train",
        tile_size: int = 512,
        processed_dataset_path: str | None = None,
    ) -> None:

        self.split = split
        self.tile_size = tile_size

        if processed_dataset_path is None:
            self.processed_dataset_path = os.path.join(
                os.getcwd(),
                "data",
                "processed",
                "whu_building",
            )
        else:
            self.processed_dataset_path = processed_dataset_path

        self.split_dir = os.path.join(
            self.processed_dataset_path,
            self.split,
        )

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(
                f"Dataset directory not found:\n"
                f"{self.split_dir}"
            )

        # Store ONLY filenames.
        # Do NOT load tensors here.
        self.files = sorted(
            filename
            for filename in os.listdir(self.split_dir)
            if filename.endswith(".pt")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .pt files found in:\n"
                f"{self.split_dir}"
            )

        print(
            f"[INFO] Found {len(self.files)} samples "
            f"for {split} split."
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        file_path = os.path.join(
            self.split_dir,
            self.files[idx],
        )

        # Load ONE sample at a time.
        image_tensor, mask_tensor = torch.load(file_path, map_location="cpu", weights_only=True)

        # Masks must already be exactly binary (0.0 or 1.0).
        # Fail loudly instead of silently "repairing" bad files, so a
        # preprocessing bug can never turn into all-empty training masks.
        mask_max = mask_tensor.max().item()
        mask_min = mask_tensor.min().item()

        if mask_min < 0.0 or mask_max not in (0.0, 1.0):
            raise ValueError(
                f"Invalid mask values in {file_path}: "
                f"min={mask_min}, max={mask_max}. "
                f"Expected a binary mask with values 0.0 / 1.0. "
                f"Re-run preprocessing."
            )

        return image_tensor, mask_tensor.float()