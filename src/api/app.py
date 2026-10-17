import io
import base64
from dotenv import load_dotenv

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile,  HTTPException, Request

from ..inference.predictor import SegmentationPredictor
from ..code.preprocessing import preprocess_transforms

load_dotenv()
app: FastAPI = FastAPI()

# Load the ensemble model
predictor: SegmentationPredictor = SegmentationPredictor.from_checkpoints(["unet", "segformer"])

@app.get("/")
def index() -> dict[str, int | str]:
    return {
        "status": 200,
        "message": "The server is running properly"
    }

@app.get("/uploads")
def get_posts_home() -> dict[str, str | int]:
    return {
        "status": 200,
        "message": "The Post are ready for the upload"
    }

@app.post("/uploads")
async def upload_post(request: Request, file: UploadFile = File(...)):
    # validating the user input

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Uploaded File extension is not supported")

    # reading the image bytewise
    image_bytes = await file.read()

    input_buffers = io.BytesIO(image_bytes)

    image = Image.open(input_buffers).convert("RGB")

    image_array = np.array(image)

    # image preprocessed

    image_preprocessed = preprocess_transforms(image_array)

    image_tensor = torch.tensor(image_preprocessed).unsqueeze(0)
    
    # Perform inference
    probs = predictor.predict_each(image_tensor)
    
    # Convert to segmentation mask

    mask_tensor = probs["ensemble"][0]
    
    # Keep mask as a numpy array for OpenCV/PIL
    mask = (mask_tensor * 255).cpu().numpy().astype(np.uint8)
    mask = Image.fromarray(mask)

    # If 'image' is already a PIL Image (e.g., loaded via Image.open()), 
    # you don't need permute/cpu/numpy. Just ensure it's RGB:
    if isinstance(image, Image.Image):
        input_image = image.convert("RGB")
    else:
        # If 'image' is a PyTorch tensor [B, C, H, W]
        input_image = image[0].permute(1, 2, 0).cpu().numpy()
        input_image = (input_image * 255).astype(np.uint8)
        input_image = Image.fromarray(input_image)

    # Resize mask to match input dimensions
    mask = mask.resize(input_image.size, Image.NEAREST)

    # Convert both to numpy arrays for easy color blending
    input_np = np.array(input_image.convert("RGB"))
    mask_np = np.array(mask)

    # 1. Choose your overlay color (RGB format)
    # [255, 0, 0] = Red, [0, 255, 0] = Green, [0, 0, 255] = Blue
    overlay_color = np.array([255, 0, 0], dtype=np.uint8) 
    
    # 2. Set transparency (0.0 = fully transparent, 1.0 = solid color)
    alpha = 0.4  

    # Find where the mask is active (assuming values > 0 are your objects)
    mask_indices = mask_np > 0

    # Blend the color into the image pixels where the mask is active
    input_np[mask_indices] = (
        (1 - alpha) * input_np[mask_indices] + alpha * overlay_color
    ).astype(np.uint8)

    # Convert back to a PIL Image
    combined = Image.fromarray(input_np)

    # Display non-blockingly on your screen
    combined.show() 

    # Save to BytesIO and encode to base64 for the API response
    combined_buffer = io.BytesIO()
    combined.save(combined_buffer, format="PNG")
    encoded_string = base64.b64encode(combined_buffer.getvalue()).decode()

    return {
        "status": 200,
        "message": "The Image has been uploaded successfully",
        "image": f"data:image/png;base64,{encoded_string}"
    }