import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import RandomInstanceDataset
from lightning_module import MultiTaskImageEncoder

def main():
    # --- Configuration ---
    IMAGE_DIR = "path/to/your/images" # IMPORTANT: User must change this path
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 100
    
    # --- Data Preparation ---
    # Check if the image directory exists
    if not os.path.isdir(IMAGE_DIR) or not os.listdir(IMAGE_DIR):
        print(f"Error: Image directory not found or empty: {IMAGE_DIR}")
        print("Please create a directory of images and update the IMAGE_DIR variable in train.py")
        # As a fallback, create a dummy image for demonstration
        os.makedirs(IMAGE_DIR, exist_ok=True)
        from PIL import Image
        dummy_img = Image.new('RGB', (128, 128), color = 'red')
        dummy_img.save(os.path.join(IMAGE_DIR, 'dummy.png'))
        print(f"Created a dummy image at {os.path.join(IMAGE_DIR, 'dummy.png')}")

    image_paths = [os.path.join(IMAGE_DIR, img_name) for img_name in os.listdir(IMAGE_DIR)]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = RandomInstanceDataset(image_paths=image_paths, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # --- Model & Trainer ---
    model = MultiTaskImageEncoder()
    logger = TensorBoardLogger("tb_logs", name="image_encoder_v1")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        gpus=1 if torch.cuda.is_available() else 0
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.fit(model, train_loader)
    print("Training complete.")

if __name__ == '__main__':
    main()
