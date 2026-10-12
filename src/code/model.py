import torch
import segmentation_models_pytorch as smp

class BuildingSegmentationModel(torch.nn.Module):
    """
    U-Net model for building segmentation.

    Encoder:
        ResNet18 pretrained on ImageNet

    Input:
        [B, 3, H, W]

    Output:
        [B, 1, H, W]
    """

    def __init__(self) -> None:
        super().__init__()

        self.model: smp.Unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)