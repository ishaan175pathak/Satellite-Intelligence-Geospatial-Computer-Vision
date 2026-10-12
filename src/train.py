import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .code.dataset import WHUBuildingDataset
from .code.model import BuildingSegmentationModel
from .code.loss import BCEDiceLoss
from .code.trainer import Trainer


def main() -> None:

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 2  # Increased accumulation steps

    # loading the dataset
    train_dataset = WHUBuildingDataset(split="train", tile_size=512)
    val_dataset = WHUBuildingDataset(split="val", tile_size=512)


    epochs: int = 50  # Increased number of epochs

    data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")

    # Model
    model: BuildingSegmentationModel = BuildingSegmentationModel().to(device)

    # Loss
    criterion: BCEDiceLoss = BCEDiceLoss()

    # Optimizerva
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Increased learning rate

    # Trainer
    trainer: Trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, accumulation_steps=ACCUMULATION_STEPS)

    # Create CSV file for logging
    log_file: str = "unet_training_log.csv"
    with open(log_file, "w") as f:
        f.write("Epoch, Train Loss, Train Accuracy, Val Loss, Val Accuracy, Val Dice, Val IoU\n")


    # Training
    best_val_loss: float = float("inf")
    best_model_path: str = "unet_best_model.pth"

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