import torch
import segmentation_models_pytorch as smp

class BuildingSegmentationModel(torch.nn.Module):
    """
    U-Net model for building segmentation and polygon extraction.

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

    def extract_polygons(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Convert segmentation masks into geospatial polygons using a basic connected components algorithm.
        """

        # Convert masks to binary (assuming 1 is building, 0 is background)
        binary_masks = (masks > 0.5).byte()

        # Initialize output list for polygons
        polygons = []

        # Get the shape of the masks
        height, width = binary_masks.shape[-2:]

        # Create a visited matrix to track visited pixels
        visited = torch.zeros((height, width), dtype=torch.bool, device=masks.device)

        # Directions for 4-connected neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Iterate over each pixel in the mask
        for y in range(height):
            for x in range(width):
                if binary_masks[y, x] == 1 and not visited[y, x]:
                    # Start a new connected component
                    component = []
                    queue = [(y, x)]
                    visited[y, x] = True

                    while queue:
                        current_y, current_x = queue.pop(0)
                        component.append((current_y, current_x))

                        # Check all four directions
                        for dy, dx in directions:
                            ny, nx = current_y + dy, current_x + dx

                            # Check if the new position is within bounds and not visited
                            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and binary_masks[ny, nx] == 1:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                    # Convert component to a tensor and add to polygons
                    polygons.append(torch.tensor(component, device=masks.device))

        return torch.stack(polygons)

