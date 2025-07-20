from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import torch

class RandomInstanceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform # To convert images to tensors, etc.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img) # Should resize to 128x128

        # Generate random bounding box [x, y, w, h] normalized
        w, h = np.random.rand(2) * 0.5 + 0.2 # width/height between 20% and 70%
        x, y = np.random.rand(2) * (1 - np.array([w, h]))
        bbox = torch.tensor([x, y, w, h])

        # Generate a random polygon segmentation mask
        mask_img = Image.new('L', (128, 128), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # 1. Randomly select number of points
        num_points = np.random.randint(4, 129)
        
        # 2. Generate points in a star-shaped pattern to ensure a valid polygon
        center_x, center_y = np.random.randint(20, 108), np.random.randint(20, 108)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
        radii = np.random.uniform(10, 60, num_points)
        
        points = []
        for angle, radius in zip(angles, radii):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((max(0, min(127, x)), max(0, min(127, y))))
            
        # 3. Draw the polygon
        draw.polygon(points, outline=1, fill=1)
        
        # 4. Convert mask to tensor
        mask = torch.from_numpy(np.array(mask_img)).unsqueeze(0).float()

        return img, bbox, mask
