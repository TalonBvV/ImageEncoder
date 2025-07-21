import os
import numpy as np
from PIL import Image
import argparse

def get_best_resize_dim(img_dim, target_dims):
    """
    Find the best target dimension for an image dimension.
    """
    # Find the target dimension that is closest to the image dimension
    return min(target_dims, key=lambda x: abs(x - img_dim))

def resize_and_patch(image_path, preprocessed_dir, patch_size, target_dims):
    """
    Resize an image, generate patches, and save them.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size

        # Get the best resize dimensions
        best_w = get_best_resize_dim(w, target_dims)
        best_h = get_best_resize_dim(h, target_dims)

        # Resize the image
        resized_img = img.resize((best_w, best_h))

        # Generate and save patches
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        for i in range(0, best_w, patch_size):
            for j in range(0, best_h, patch_size):
                if i + patch_size <= best_w and j + patch_size <= best_h:
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

    target_dims = [256, 384, 512, 640, 768, 896, 1024]
    patch_size = 128

    image_paths = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    for image_path in image_paths:
        resize_and_patch(image_path, args.preprocessed_dir, patch_size, target_dims)

    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
