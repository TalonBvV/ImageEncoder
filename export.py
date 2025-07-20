import torch
from lightning_module import MultiTaskImageEncoder

def export_encoder_to_onnx(checkpoint_path, export_path="encoder.onnx"):
    """
    Loads the trained encoder from a checkpoint and exports it to ONNX format.
    """
    # Load the full model from the checkpoint
    full_model = MultiTaskImageEncoder.load_from_checkpoint(checkpoint_path)
    
    # Extract the encoder
    encoder = full_model.encoder
    encoder.eval() # Set the model to evaluation mode
    
    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 3, 128, 128)
    
    # Export the model
    print(f"Exporting encoder to {export_path}...")
    torch.onnx.export(
        encoder,
        dummy_input,
        export_path,
        input_names=['input'],
        output_names=['latent_vector'],
        dynamic_axes={'input': {0: 'batch_size'}, 'latent_vector': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == '__main__':
    # IMPORTANT: User must provide the path to their trained model checkpoint
    # Example:
    # CHECKPOINT_PATH = "tb_logs/image_encoder_v1/version_0/checkpoints/epoch=49-step=150.ckpt"
    # export_encoder_to_onnx(CHECKPOINT_PATH)
    
    print("To export the model, you must provide the path to a trained checkpoint.")
    print("Example usage:")
    print("CHECKPOINT_PATH = \"path/to/your/checkpoint.ckpt\"")
    print("export_encoder_to_onnx(CHECKPOINT_PATH)")
