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
    PREPROCESSED_DIR = "preprocessed_data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 100
    
    # --- Preprocessing ---
    print("Starting preprocessing...")
    os.system(f"python preprocess.py --image_dir {IMAGE_DIR} --preprocessed_dir {PREPROCESSED_DIR}")
    print("Preprocessing finished.")

    # --- Data Preparation ---
    patch_paths = []
    for root, _, files in os.walk(PREPROCESSED_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                patch_paths.append(os.path.join(root, file))

    if not patch_paths:
        print(f"Error: No preprocessed images found in {PREPROCESSED_DIR}")
        print("Please ensure the preprocessing step ran correctly and generated patch images.")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = RandomInstanceDataset(patch_paths=patch_paths, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # --- Model & Trainer ---
    model = MultiTaskImageEncoder()
    logger = TensorBoardLogger("tb_logs", name="image_encoder_v1")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        accelerator="auto",
        devices="auto"
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.fit(model, train_loader)
    print("Training complete.")

if __name__ == '__main__':
    main()
