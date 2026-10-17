FROM python:3.12-slim

# Set environment variables for clean logs and performance optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app

WORKDIR /app

# 1. Copy the requirements file first so it is available for the install layer
COPY requirements.txt .

# 2. Install computer vision runtime libraries, upgrade pip, 
#    and install packages pulling PyTorch from its official CUDA 12.8 repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy your application source code and pre-trained weights
COPY src/ ./src/
COPY unet_best_model.pth .
COPY segformer_mit-b0_best_model.pth .

# Expose port and launch the FastAPI web app via Uvicorn
EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]