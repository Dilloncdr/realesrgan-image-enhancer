import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def enhance_images(input_folder, output_folder, scale=4):
    """
    Enhance all images inside input_folder using Real-ESRGAN.
    Results are saved in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")

    # Path to the pretrained model
    MODEL_PATH = os.path.join("models", "RealESRGAN_x4plus.pth")

    # Define ESRGAN model
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    # Adjust settings for GPU VRAM limits
    tile_size = 200 if device == 'cuda' else 0

    # Create Real-ESRGAN upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=MODEL_PATH,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=(device == 'cuda'),  # use FP16 on GPU for speed
        device=device
    )

    print("✅ Model loaded successfully.\n")

    # Get all supported images
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    if not files:
        print("❌ No images found in input folder.")
        return

    # Process all images with progress bar
    for filename in tqdm(files, desc="✨ Enhancing images", unit="img"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Open using Pillow to handle non-ASCII (Persian) filenames
            img = Image.open(input_path).convert('RGB')
            np_img = np.array(img)

            # Enhance the image
            output, _ = upsampler.enhance(np_img, outscale=scale)

            # Save output (again using Pillow to support Unicode filenames)
            Image.fromarray(output).save(output_path)

        except Exception as e:
            print(f"\n⚠️ Failed to enhance {filename}: {e}")

    print(f"\n🎉 Done! Enhanced images are saved in:\n👉 {output_folder}")


if __name__ == "__main__":
    INPUT_FOLDER = "input"
    OUTPUT_FOLDER = "output"

    enhance_images(INPUT_FOLDER, OUTPUT_FOLDER, scale=4)
