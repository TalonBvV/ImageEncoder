import os
import numpy as np
from PIL import Image
import argparse

def resize_and_patch(image_path, preprocessed_dir, patch_size, target_size):
    """
    Resize an image to a target size, generate patches, and save them.
    """
    try:
        img = Image.open(image_path).convert('RGB')

        # Resize the image
        resized_img = img.resize(target_size)
        w, h = resized_img.size

        # Generate and save patches
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        for i in range(0, w, patch_size):
            for j in range(0, h, patch_size):
                if i + patch_size <= w and j + patch_size <= h:
                    patch = resized_img.crop((i, j, i + patch_size, j + patch_size))
                    patch_name = f"{img_name}_patch_{i}_{j}.png"
                    patch.save(os.path.join(preprocessed_dir, patch_name))
    except Exception as e:
        print(f"Could not process {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess images for training.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with original images.")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory to save preprocessed patches.")
    args = parser.parse_args()

    os.makedirs(args.preprocessed_dir, exist_ok=True)

    target_size = (512, 512)
    patch_size = 128

    image_paths = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    for image_path in image_paths:
        resize_and_patch(image_path, args.preprocessed_dir, patch_size, target_size)

    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
