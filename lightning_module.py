import torch
import pytorch_lightning as pl
from models.encoder import ImageEncoder
from models.decoder import ImageDecoder
import torch.nn.functional as F
import torchvision

class MultiTaskImageEncoder(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = ImageEncoder(latent_dim=384)
        
        # --- Decoders ---
        # Head 1: No conditioning
        self.decoder_recon = ImageDecoder(latent_dim=384, conditioning_dim=0)
        # Head 2: Conditioned on 4 bbox coordinates
        self.decoder_bbox = ImageDecoder(latent_dim=384, conditioning_dim=4)
        # Head 3: Conditioned on a downsampled 16x16=256 segmentation mask
        self.decoder_seg = ImageDecoder(latent_dim=384, conditioning_dim=256)
        
        # Layer to downsample the segmentation mask for conditioning
        self.seg_mask_downsampler = torch.nn.AdaptiveAvgPool2d((16, 16))

        # Loss weights
        self.w_recon = 1.0
        self.w_bbox = 1.0
        self.w_seg = 1.0

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        image, bbox, seg_mask = batch
        
        # 1. Get latent vector
        latent_vector = self.encoder(image)
        
        # --- Task 1: Full Reconstruction ---
        reconstructed_image = self.decoder_recon(latent_vector)
        loss_recon = F.mse_loss(reconstructed_image, image)
        
        # --- Task 2: Bounding Box Reconstruction ---
        reconstructed_bbox_image = self.decoder_bbox(latent_vector, conditioning=bbox)
        # We still calculate the loss over the whole image, but the decoder
        # now has the bbox context to know which area to focus on.
        loss_bbox = F.mse_loss(reconstructed_bbox_image, image)

        # --- Task 3: Segmentation Reconstruction ---
        # Downsample the segmentation mask to create the conditioning vector
        seg_conditioning = self.seg_mask_downsampler(seg_mask).view(image.size(0), -1)
        reconstructed_seg_image = self.decoder_seg(latent_vector, conditioning=seg_conditioning)
        
        # The loss is still only applied to the masked region
        masked_output = reconstructed_seg_image * seg_mask
        masked_original = image * seg_mask
        loss_seg = F.mse_loss(masked_output, masked_original)
        
        # --- Total Loss ---
        total_loss = (self.w_recon * loss_recon + 
                      self.w_bbox * loss_bbox + 
                      self.w_seg * loss_seg)
        
        # --- Logging ---
        self.log_dict({
            'train_loss/total': total_loss,
            'train_loss/recon': loss_recon,
            'train_loss/bbox': loss_bbox,
            'train_loss/seg': loss_seg
        })
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: # Only log images for the first batch
            image, bbox, seg_mask = batch
            latent = self.encoder(image)
            
            # Get reconstructions
            recon_full = self.decoder_recon(latent)
            
            seg_conditioning = self.seg_mask_downsampler(seg_mask).view(image.size(0), -1)
            recon_seg_conditioned = self.decoder_seg(latent, conditioning=seg_conditioning)
            
            # Apply mask for visualization
            masked_output = recon_seg_conditioned * seg_mask
            
            grid = torchvision.utils.make_grid([image[0], recon_full[0], masked_output[0]])
            self.logger.experiment.add_image('val_samples', grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
