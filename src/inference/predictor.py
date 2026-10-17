"""Model-agnostic inference helpers for the WHU building models.

Loads a trained U-Net and/or SegFormer checkpoint and turns preprocessed
image tiles into per-pixel building probabilities. When more than one model
is loaded, an "ensemble" (mean of the probabilities) is produced as well.

Run the scripts that use this module from the repository root, as modules:

    python -m src.geospatial.evaluate_polygons --arch unet --split val
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ARCHS: Tuple[str, ...] = ("unet", "segformer")
ARCH_CHOICES: Tuple[str, ...] = (*ARCHS, "ensemble")

DEFAULT_CHECKPOINTS: Dict[str, str] = {
    "unet": "unet_best_model.pth",
    "segformer": "segformer_mit-b0_best_model.pth",
}

# Tiles are stored ImageNet-normalized (see preprocessing.py).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Normalized [3, H, W] tensor -> displayable [H, W, 3] float array."""

    image = image.detach().cpu().float()

    return (
        (image * IMAGENET_STD + IMAGENET_MEAN)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def build_model(arch: str, segformer_name: str = "nvidia/mit-b0") -> torch.nn.Module:
    """Create an (untrained) model. Imports are lazy so the U-Net does not
    need `transformers`, and SegFormer does not need `segmentation_models_pytorch`."""

    if arch == "unet":
        from ..code.model import BuildingSegmentationModel

        return BuildingSegmentationModel()

    if arch == "segformer":
        from ..code.segformer import SegFormerBuildingModel

        return SegFormerBuildingModel(segformer_name)

    raise ValueError(f"Unknown architecture: {arch!r} (expected one of {ARCHS})")


def load_model(
    arch: str,
    checkpoint_path: str,
    device: torch.device,
    segformer_name: str = "nvidia/mit-b0",
) -> torch.nn.Module:
    """Build ``arch`` and load a state_dict saved by ``train.py``."""

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    model = build_model(arch, segformer_name).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


class SegmentationPredictor:
    """Runs one or more segmentation models and returns probabilities.

    ``predict_each`` returns ``{name: probs}`` with ``probs`` of shape
    [B, H, W] (float32, CPU, values in [0, 1]). With two or more models an
    extra ``"ensemble"`` entry holds the mean probability.
    """

    def __init__(
        self,
        models: Mapping[str, torch.nn.Module],
        device: Optional[torch.device] = None,
    ) -> None:

        if not models:
            raise ValueError("At least one model is required.")

        self.device = device or default_device()
        self.use_amp = self.device.type == "cuda"

        self.models: Dict[str, torch.nn.Module] = {
            name: model.to(self.device).eval()
            for name, model in models.items()
        }

    @classmethod
    def from_checkpoints(
        cls,
        archs: Sequence[str],
        checkpoints: Optional[Mapping[str, str]] = None,
        device: Optional[torch.device] = None,
        segformer_name: str = "nvidia/mit-b0",
    ) -> "SegmentationPredictor":

        device = device or default_device()
        checkpoints = dict(checkpoints or {})

        models: Dict[str, torch.nn.Module] = {}

        for arch in archs:
            path = checkpoints.get(arch) or os.path.join(
                os.getcwd(), DEFAULT_CHECKPOINTS[arch]
            )
            print(path)
            print(f"[INFO] Loading {arch} from {path}")
            models[arch] = load_model(arch, path, device, segformer_name)

        return cls(models, device)

    @property
    def names(self) -> List[str]:
        names = list(self.models)

        if len(names) > 1:
            names.append("ensemble")

        return names

    @torch.inference_mode()
    def predict_each(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """images: normalized [B, 3, H, W] -> {name: [B, H, W] probabilities}."""

        images = images.to(self.device, non_blocking=True)

        outputs: Dict[str, torch.Tensor] = {}

        for name, model in self.models.items():
            # Same mixed-precision setup as training / validation.
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits = model(images)

            outputs[name] = torch.sigmoid(logits.float())[:, 0].cpu()

        members = list(outputs.values())

        if len(members) > 1:
            outputs["ensemble"] = torch.stack(members).mean(dim=0)

        return outputs


def iter_split_batches(
    dataset,
    batch_size: int = 8,
) -> Iterator[Tuple[List[str], torch.Tensor, torch.Tensor]]:
    """Yield ``(files, images [B,3,H,W], masks [B,H,W])`` in dataset order."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    start = 0

    for images, masks in loader:
        count = images.size(0)
        files = list(dataset.files[start : start + count])
        start += count

        yield files, images, masks


def iter_split_probabilities(
    predictor: SegmentationPredictor,
    dataset,
    batch_size: int = 8,
    desc: str = "Predicting",
) -> Iterator[Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]]:
    """Yield ``(files, gt_masks [B,H,W] bool, {name: probs [B,H,W] float32})``."""

    from tqdm import tqdm

    total = math.ceil(len(dataset) / batch_size)

    for files, images, masks in tqdm(
        iter_split_batches(dataset, batch_size),
        total=total,
        desc=desc,
    ):
        probs = predictor.predict_each(images)

        yield (
            files,
            masks.numpy() > 0.5,
            {name: p.numpy() for name, p in probs.items()},
        )


def add_predictor_args(
    parser: argparse.ArgumentParser,
    extra_archs: Sequence[str] = (),
) -> None:
    """Shared CLI options for choosing the model(s) and checkpoint paths."""

    parser.add_argument(
        "--arch",
        choices=[*ARCH_CHOICES, *extra_archs],
        default="unet",
        help="Model to run. 'ensemble' averages the U-Net and SegFormer "
        "probabilities.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        default=os.path.join(os.getcwd(), DEFAULT_CHECKPOINTS["unet"]),
    )
    parser.add_argument(
        "--segformer-checkpoint",
        default=os.path.join(os.getcwd(), DEFAULT_CHECKPOINTS["segformer"]),
    )
    parser.add_argument("--segformer-name", default="nvidia/mit-b0")


def predictor_from_args(args: argparse.Namespace) -> SegmentationPredictor:
    """Build the predictor selected by ``--arch`` (loads both for ensemble)."""

    archs = list(ARCHS) if args.arch == "ensemble" else [args.arch]

    return SegmentationPredictor.from_checkpoints(
        archs,
        {
            "unet": args.unet_checkpoint,
            "segformer": args.segformer_checkpoint,
        },
        segformer_name=args.segformer_name,
    )
