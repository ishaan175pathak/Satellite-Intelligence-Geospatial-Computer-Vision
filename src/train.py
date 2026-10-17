import argparse
import os
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .code.dataset import WHUBuildingDataset
from .code.model import BuildingSegmentationModel
from .code.loss import BCEDiceLoss
from .code.trainer import Trainer


def build_model(name: str, segformer_name: str) -> torch.nn.Module:

    if name == "unet":
        return BuildingSegmentationModel()

    if name == "segformer":
        # Imported lazily so the U-Net does not require `transformers`.
        from .code.segformer import SegFormerBuildingModel

        return SegFormerBuildingModel(segformer_name)

    raise ValueError(f"Unknown model: {name}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train a building segmentation model on WHU."
    )
    parser.add_argument(
        "--model", choices=["unet", "segformer"], default="unet"
    )
    parser.add_argument(
        "--segformer-name",
        default="nvidia/mit-b0",
        help="Hugging Face checkpoint, e.g. nvidia/mit-b0 or nvidia/mit-b2",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing log / checkpoint.",
    )

    return parser.parse_args()


def main() -> None:

    load_dotenv()
    args = parse_args()

    # Same seed for every model so comparisons are fair.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output names are per model, so one run never overwrites another.
    if args.model == "unet":
        prefix = "unet"
    else:
        prefix = f"segformer_{args.segformer_name.split('/')[-1]}"

    log_file: str = f"{prefix}_training_log.csv"
    best_model_path: str = f"{prefix}_best_model.pth"

    existing = [
        p for p in (log_file, best_model_path) if os.path.exists(p)
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {existing}. Move or back them up, "
            f"or pass --overwrite."
        )

    print(
        f"Model: {args.model} | lr: {args.lr} | epochs: {args.epochs} | "
        f"seed: {args.seed}"
    )
    print(f"Log: {log_file} | Checkpoint: {best_model_path}")

    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 2

    # loading the dataset
    train_dataset = WHUBuildingDataset(split="train", tile_size=512)
    val_dataset = WHUBuildingDataset(split="val", tile_size=512)

    epochs: int = args.epochs

    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Model
    model: torch.nn.Module = build_model(
        args.model, args.segformer_name
    ).to(device)

    # Loss
    criterion: BCEDiceLoss = BCEDiceLoss()

    # Optimizer
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Trainer
    trainer: Trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, accumulation_steps=ACCUMULATION_STEPS)

    # Create CSV file for logging
    with open(log_file, "w") as f:
        f.write("Epoch, Train Loss, Train Accuracy, Val Loss, Val Accuracy, Val Dice, Val IoU\n")


    # Training
    best_val_loss: float = float("inf")

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = trainer.train_epoch(data_loader)

        val_loss, val_acc, val_dice, val_iou = (trainer.validate_epoch(val_loader))

        # Log metrics to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{train_loss:.4f},{train_acc:.4f}, {val_loss:.4f},{val_acc:.4f}, {val_dice:.4f},{val_iou:.4f}\n")

        print(
            f"Epoch {epoch + 1} "
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_acc:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_acc:.4f} | "
            f"Validation Dice: {val_dice:.4f} | "
            f"Validation IoU: {val_iou:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()