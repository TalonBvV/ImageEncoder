from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import torch

class RandomInstanceDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform # To convert images to tensors, etc.

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.patch_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Generate random bounding box [x, y, w, h] normalized
        w, h = np.random.rand(2) * 0.5 + 0.2 # width/height between 20% and 70%
        x, y = np.random.rand(2) * (1 - np.array([w, h]))
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        # --- Generate a more complex and random segmentation mask ---
        mask_img = Image.new('L', (128, 128), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # Randomly choose a shape generation method
        shape_type = np.random.choice(['polygons', 'ellipses', 'blob'])

        if shape_type == 'polygons':
            # Draw 2 to 5 smaller, overlapping polygons
            num_polygons = np.random.randint(2, 6)
            for _ in range(num_polygons):
                num_points = np.random.randint(3, 10)
                center_x, center_y = np.random.randint(10, 118), np.random.randint(10, 118)
                angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
                radii = np.random.uniform(5, 30, num_points)
                points = []
                for angle, radius in zip(angles, radii):
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    points.append((max(0, min(127, x)), max(0, min(127, y))))
                draw.polygon(points, outline=1, fill=1)

        elif shape_type == 'ellipses':
            # Draw 2 to 5 random ellipses
            num_ellipses = np.random.randint(2, 6)
            for _ in range(num_ellipses):
                x1 = np.random.randint(0, 100)
                y1 = np.random.randint(0, 100)
                x2 = x1 + np.random.randint(10, 50)
                y2 = y1 + np.random.randint(10, 50)
                draw.ellipse([x1, y1, x2, y2], outline=1, fill=1)
                
        else: # 'blob'
            # Generate a single, more complex and irregular polygon
            num_points = np.random.randint(10, 40)
            center_x, center_y = np.random.randint(30, 98), np.random.randint(30, 98)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
            # Add more variance to the radii to create a "blobbier" shape
            radii = np.random.uniform(20, 60, num_points) + np.random.normal(0, 10, num_points)
            
            points = []
            for angle, radius in zip(angles, radii):
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                points.append((max(0, min(127, x)), max(0, min(127, y))))
            draw.polygon(points, outline=1, fill=1)

        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask_img)).unsqueeze(0).float()

        return img, bbox, mask
