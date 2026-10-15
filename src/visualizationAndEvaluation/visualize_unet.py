import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..code.dataset import WHUBuildingDataset
from ..code.model import BuildingSegmentationModel


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Error overlay colors (RGB, 0 to 1).
GREEN = np.array([0.15, 0.80, 0.25])  # correct building (TP)
RED = np.array([0.90, 0.15, 0.15])    # false alarm (FP)
BLUE = np.array([0.20, 0.45, 1.00])   # missed building (FN)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def load_model(
    model_path: str,
    device: torch.device,
) -> BuildingSegmentationModel:

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model = BuildingSegmentationModel().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset: WHUBuildingDataset,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> List[Dict]:
    """Run the model on every tile and return per-tile pixel counts
    and scores. Order matches dataset.files."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    use_amp = device.type == "cuda"
    dims = (1, 2, 3)

    rows: List[Dict] = []
    tile_index = 0

    for images, masks in tqdm(loader, desc="Scoring test tiles"):
        images = images.to(device, non_blocking=True)
        targets = masks.unsqueeze(1).to(device) > 0.5

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)

        preds = torch.sigmoid(logits.float()) > threshold

        tp = (preds & targets).sum(dim=dims).cpu().tolist()
        fp = (preds & ~targets).sum(dim=dims).cpu().tolist()
        fn = (~preds & targets).sum(dim=dims).cpu().tolist()

        for b in range(images.size(0)):
            gt_pixels = tp[b] + fn[b]
            pred_pixels = tp[b] + fp[b]

            # IoU is only defined when the tile has buildings.
            iou: Optional[float] = (
                safe_div(tp[b], tp[b] + fp[b] + fn[b])
                if gt_pixels > 0
                else None
            )

            rows.append(
                {
                    "index": tile_index,
                    "file": dataset.files[tile_index],
                    "gt_pixels": gt_pixels,
                    "pred_pixels": pred_pixels,
                    "iou": iou,
                }
            )
            tile_index += 1

    return rows


def select_tiles(
    rows: List[Dict],
    n_best: int,
    n_median: int,
    n_worst: int,
    n_empty: int,
) -> List[Tuple[str, Dict]]:
    """Pick tiles by per-tile IoU so the figure shows the full range
    of behavior instead of the first few (mostly empty) tiles."""

    with_buildings = sorted(
        (r for r in rows if r["gt_pixels"] > 0),
        key=lambda r: r["iou"],
    )

    selected: List[Tuple[str, Dict]] = []

    if n_best > 0:
        for r in with_buildings[-n_best:][::-1]:
            selected.append(("Best", r))

    if n_median > 0 and with_buildings:
        start = max(len(with_buildings) // 2 - n_median // 2, 0)
        for r in with_buildings[start:start + n_median]:
            selected.append(("Median", r))

    if n_worst > 0:
        for r in with_buildings[:n_worst]:
            selected.append(("Worst", r))

    if n_empty > 0:
        empty = sorted(
            (r for r in rows if r["gt_pixels"] == 0),
            key=lambda r: r["pred_pixels"],
            reverse=True,
        )
        for r in empty[:n_empty]:
            selected.append(("Empty tile", r))

    # Drop duplicates (possible on very small datasets).
    seen = set()
    unique: List[Tuple[str, Dict]] = []
    for label, r in selected:
        if r["index"] not in seen:
            seen.add(r["index"])
            unique.append((label, r))

    return unique


def error_overlay(
    display_image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:

    overlay = display_image * 0.35

    overlay[target & prediction] = GREEN
    overlay[~target & prediction] = RED
    overlay[target & ~prediction] = BLUE

    return overlay


def plot_learning_curves(
    log_path: str,
    output_dir: str = "results/visualizations",
) -> None:
    """Plot loss, accuracy and val Dice/IoU from the training CSV."""

    if not os.path.isfile(log_path):
        print(f"[WARNING] Training log not found, skipping curves: {log_path}")
        return

    # The logger writes spaces after the commas, so strip every header
    # and value instead of relying on exact column names.
    with open(log_path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        records = [
            [float(v) for v in row]
            for row in reader
            if row and all(v.strip() for v in row)
        ]

    if not records:
        print(f"[WARNING] Training log is empty, skipping curves: {log_path}")
        return

    columns = {
        name: np.array([rec[i] for rec in records])
        for i, name in enumerate(header)
    }

    epochs = columns["Epoch"]
    best_loss_epoch = int(epochs[np.argmin(columns["Val Loss"])])
    best_dice_epoch = int(epochs[np.argmax(columns["Val Dice"])])

    train_color, val_color = "#2a6fdb", "#e8821e"
    dice_color, iou_color = "#2a9d5b", "#8a4fd6"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, columns["Train Loss"], color=train_color, label="Train")
    axes[0].plot(epochs, columns["Val Loss"], color=val_color, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("BCE + Dice loss")

    axes[1].plot(epochs, columns["Train Accuracy"], color=train_color, label="Train")
    axes[1].plot(epochs, columns["Val Accuracy"], color=val_color, label="Validation")
    axes[1].set_title("Pixel accuracy")
    axes[1].set_ylabel("Accuracy")

    axes[2].plot(epochs, columns["Val Dice"], color=dice_color, label="Val Dice")
    axes[2].plot(epochs, columns["Val IoU"], color=iou_color, label="Val IoU")
    axes[2].set_title("Validation Dice and IoU")
    axes[2].set_ylabel("Score")

    for ax in axes:
        ax.axvline(
            best_loss_epoch,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Saved checkpoint (epoch {best_loss_epoch})",
        )
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[INFO] Epochs: {len(epochs)} | "
        f"lowest val loss at epoch {best_loss_epoch} "
        f"(Dice {columns['Val Dice'][best_loss_epoch - 1]:.4f}) | "
        f"best val Dice {columns['Val Dice'].max():.4f} "
        f"at epoch {best_dice_epoch}"
    )
    print(f"[INFO] Saved learning curves: {output_path}")


@torch.inference_mode()
def visualize_predictions(
    model_path: str,
    output_dir: str = "results/visualizations",
    threshold: float = 0.5,
    n_best: int = 2,
    n_median: int = 2,
    n_worst: int = 2,
    n_empty: int = 1,
    batch_size: int = 8,
) -> None:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(model_path, device)

    dataset = WHUBuildingDataset(split="test", tile_size=512)

    rows = evaluate_dataset(
        model,
        dataset,
        device,
        threshold=threshold,
        batch_size=batch_size,
    )

    picks = select_tiles(rows, n_best, n_median, n_worst, n_empty)

    if not picks:
        raise RuntimeError("No tiles selected for visualization.")

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        len(picks),
        4,
        figsize=(16, len(picks) * 4),
        squeeze=False,
    )

    for row_idx, (label, r) in enumerate(picks):

        image, mask = dataset[r["index"]]

        image = image.detach().cpu().float()
        target = (mask.detach().cpu().float().squeeze() > 0.5).numpy()

        display_image = (
            (image * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        )

        logits = model(image.unsqueeze(0).to(device))
        prediction = (
            torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold
        )

        score = (
            f"IoU {r['iou']:.3f}"
            if r["iou"] is not None
            else f"false-positive px: {r['pred_pixels']}"
        )

        axes[row_idx, 0].imshow(display_image)
        axes[row_idx, 0].set_title(
            f"{label}: {r['file'][:-3]}", fontsize=9
        )

        axes[row_idx, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=9)

        axes[row_idx, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f"Prediction ({score})", fontsize=9)

        axes[row_idx, 3].imshow(
            error_overlay(display_image, target, prediction)
        )
        axes[row_idx, 3].set_title("Errors", fontsize=9)

        for ax in axes[row_idx]:
            ax.axis("off")

        print(
            f"{label:<10} {r['file'][:-3]} | "
            f"GT px {r['gt_pixels']} | pred px {r['pred_pixels']} | "
            f"{score}"
        )

    fig.legend(
        handles=[
            Patch(color=GREEN, label="Correct building"),
            Patch(color=RED, label="False alarm"),
            Patch(color=BLUE, label="Missed building"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))

    output_path = os.path.join(
        output_dir, "whu_test_predictions.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved visualization: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize U-Net predictions on the WHU test split."
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.getcwd(), "unet_best_model.pth"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/visualizations")
    parser.add_argument(
        "--log-path",
        default=os.path.join(os.getcwd(), "unet_training_log.csv"),
    )
    args = parser.parse_args()

    plot_learning_curves(
        log_path=args.log_path,
        output_dir=args.output_dir,
    )

    visualize_predictions(
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )