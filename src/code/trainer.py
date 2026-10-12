import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple


def segmentation_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float]:

    predictions = torch.sigmoid(logits) > threshold
    targets = masks > 0.5

    intersection = (predictions & targets).sum().item()
    predicted_pixels = predictions.sum().item()
    target_pixels = targets.sum().item()
    union = (predictions | targets).sum().item()

    eps = 1e-8

    dice = (2.0 * intersection + eps) / (
        predicted_pixels + target_pixels + eps
    )

    iou = (intersection + eps) / (union + eps)

    return dice, iou


class Trainer:
    """Handles model training and validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        accumulation_steps: int = 2,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp
        )

    def train_epoch(
        self, data_loader: DataLoader
    ) -> Tuple[float, float]:

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        total_samples = 0
        building_pixels = 0

        progress_bar = tqdm(data_loader, desc="Training")
        num_batches = len(data_loader)

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(
                self.device, non_blocking=True
            )
            masks = masks.unsqueeze(1).float().to(
                self.device, non_blocking=True
            )

            # Handle the final incomplete accumulation group.
            group_start = (
                batch_idx // self.accumulation_steps
            ) * self.accumulation_steps

            group_size = min(
                self.accumulation_steps,
                num_batches - group_start,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            # Scale each microbatch loss before accumulating
            # gradients. The final partial group is handled too.
            self.scaler.scale(loss / group_size).backward()

            # Compute metrics for EVERY training batch.
            batch_size = images.size(0)
            running_loss += loss.detach().item() * batch_size
            total_samples += batch_size

            with torch.no_grad():
                preds = torch.sigmoid(logits) > 0.5
                targets = masks > 0.5

                running_corrects += (
                    (preds == targets).sum().item()
                )
                total_pixels += targets.numel()
                building_pixels += targets.sum().item()

            # Update model after the accumulation group.
            if (
                (batch_idx + 1) % self.accumulation_steps == 0
                or (batch_idx + 1) == num_batches
            ):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{running_corrects / total_pixels:.4f}",
            )

        if building_pixels == 0:
            raise RuntimeError(
                "Training masks contain no building pixels. "
                "Check preprocessing before training."
            )

        print(
            f"[INFO] Train building pixel fraction: "
            f"{building_pixels / total_pixels:.4f}"
        )

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_pixels

        return epoch_loss, epoch_acc

    def validate_epoch(
        self, data_loader: DataLoader
    ) -> Tuple[float, float, float, float]:

        self.model.eval()

        total_intersection: int = 0
        total_predicted: float = 0
        total_targets: int = 0
        total_union: int = 0
        running_loss: float = 0.0
        running_corrects: int = 0
        total_pixels: int = 0
        total_samples: int = 0

        progress_bar = tqdm(data_loader, desc="Validation")

        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(
                    self.device, non_blocking=True
                )
                masks = masks.unsqueeze(1).float().to(
                    self.device, non_blocking=True
                )

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.sigmoid(logits) > 0.5
                targets = masks > 0.5


                total_intersection += (preds & targets).sum().item()
                total_predicted += preds.sum().item()
                total_targets += targets.sum().item()
                total_union += (preds | targets).sum().item()

                running_corrects += (
                    (preds == targets).sum().item()
                )
                total_pixels += targets.numel()

                progress_bar.set_postfix(
                    loss=f"{running_loss / total_samples:.4f}",
                    acc=f"{running_corrects / total_pixels:.4f}",
                )

        if total_targets == 0:
            raise RuntimeError(
                "Validation masks contain no building pixels, so "
                "Dice/IoU would be meaningless. "
                "Check the val split before training."
            )

        print(
            f"[INFO] Val building pixel fraction: "
            f"{total_targets / total_pixels:.4f}"
        )

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_pixels

        eps = 1e-8

        dice = (2.0 * total_intersection + eps) / (total_predicted + total_targets + eps)

        iou = (total_intersection + eps) / (total_union + eps)

        return epoch_loss, epoch_acc, dice, iou