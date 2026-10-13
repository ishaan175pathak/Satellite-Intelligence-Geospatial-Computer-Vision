import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerBuildingModel(torch.nn.Module):
    """
    SegFormer model for building segmentation.

    Same interface as BuildingSegmentationModel (the U-Net), so the
    existing loss, trainer, and evaluation code work unchanged.

    Encoder:
        Mix Transformer (MiT) pretrained on ImageNet. The decode head
        is newly initialized, so Hugging Face prints a warning about
        uninitialized weights on first load. That is expected.

    Input:
        [B, 3, H, W]  (ImageNet-normalized)

    Output:
        [B, 1, H, W]  raw logits at full input resolution
    """

    def __init__(self, pretrained_name: str = "nvidia/mit-b0") -> None:
        super().__init__()

        self.model: SegformerForSemanticSegmentation = (
            SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=1,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SegFormer returns logits at 1/4 of the input resolution.
        logits = self.model(pixel_values=x).logits

        # Upsample so the output matches the [B, 1, H, W] masks.
        return F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )