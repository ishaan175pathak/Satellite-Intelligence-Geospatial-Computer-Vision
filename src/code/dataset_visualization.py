import matplotlib.pyplot as plt
import numpy
import os
from PIL import Image
from typing import LiteralString, Tuple

# defining the path
CWD: str = os.getcwd()
RAW_DATASET_PATH: str = os.path.join(CWD, "data", "raw", "whu_building", "train")


# loading the images and masks
IMAGE_PATH: str = os.path.join(RAW_DATASET_PATH, "Image")
MASK_PATH: str = os.path.join(RAW_DATASET_PATH, "Mask")

# defining a visualization method

def visualization(image_path: str, mask_path: str, figure_size: Tuple[int, int] = (256, 256)) -> None:
    """
        This method helps in the visualization of the dataset. 
        Args:
            - image_path (str) : Absolute path of the image
            - mask_path (str) : Absolute path of the mask
            - figure_size (tuple[int, int]) : Target figure size for the image and mask
        Returns:
            None   
    """
    # Convert to numpy arrays
    image_array = numpy.array(Image.open(image_path).convert("RGB").resize(figure_size))
    mask_array = numpy.array(Image.open(mask_path).convert("L").resize(figure_size))  # Convert to grayscale

    
    # Plot the image and mask

    plt.figure(figsize=(12, 6))

    plt.suptitle("Comparison of the satellite images and building masks")
    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title("Satellite Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_array, cmap="gray")
    plt.title("Building Mask")
    plt.axis("off")
    plt.show()


# getting a random image and mask

# getting the list of images and the masks

image: str = numpy.random.choice(os.listdir(IMAGE_PATH))
mask: str = numpy.random.choice(os.listdir(MASK_PATH))

visualization(os.path.join(IMAGE_PATH, image), os.path.join(MASK_PATH, mask))