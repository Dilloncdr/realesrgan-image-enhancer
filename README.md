# Real-ESRGAN Image Enhancer (2× / 4× Super-Resolution)

Batch image upscaling and enhancement using **Real-ESRGAN** (RRDBNet backbone).  
Supports Unicode/Persian filenames via Pillow.

## What it does
- Upscales images by **2×** or **4×**
- Reduces compression artifacts and improves perceptual sharpness
- Uses GPU (CUDA) automatically if available, otherwise CPU
- Uses **tiling** on GPU to avoid VRAM out-of-memory errors

## How it works (high level)
This project loads a pretrained **Real-ESRGAN** model and runs inference on each image:
1) Read image with Pillow (Unicode-safe)
2) Convert to NumPy array
3) Run `RealESRGANer.enhance()` to generate a higher-resolution output
4) Save result to `output/`

## Folder structure
- `input/`  → put your images here
- `output/` → enhanced images are saved here
- `models/` → place model weights here (not committed)

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
