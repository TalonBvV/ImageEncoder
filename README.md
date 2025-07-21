# Multi-Task Image Encoder

This project contains a high-performance, multi-task image encoder built with PyTorch and PyTorch Lightning. The encoder is trained to create a robust 384-dimensional latent vector from a 128x128 image by learning from three simultaneous reconstruction tasks.

## Features

- **High-Performance Backbone:** Uses a pretrained `MobileVit-xs` for a strong balance of speed and accuracy.
- **Multi-Task Learning:** The encoder is trained on three objectives simultaneously:
    1.  **Full Image Reconstruction:** Recreates the entire input image.
    2.  **Conditioned Bounding Box Reconstruction:** Reconstructs the image, guided by input bounding box coordinates.
    3.  **Conditioned Segmentation Reconstruction:** Reconstructs the image, guided by an input segmentation mask.
- **Advanced Data Generation:** All bounding boxes and complex polygonal segmentation masks are generated on the fly.
- **Exportable:** The final trained encoder can be easily exported to ONNX format.
- **Colab Ready:** Includes a `setup.py` and `requirements.txt` for easy installation in any environment.

## Project Structure

```
.
├── data/
│   └── dataset.py          # Handles data loading and on-the-fly generation of tasks.
├── models/
│   ├── encoder.py          # The main ImageEncoder model.
│   └── decoder.py          # The Decoder architecture used by the learning heads.
├── image_encoder_tutorial.ipynb # Google Colab tutorial notebook.
├── lightning_module.py     # The core PyTorch Lightning module orchestrating the multi-task training.
├── train.py                # Script to run the training process locally.
├── export.py               # Script to export the trained encoder to ONNX.
├── requirements.txt        # Project dependencies.
├── setup.py                # Makes the project installable as a Python package.
└── README.md               # This file.
```

## Installation

To install the package directly from a GitHub repository, use the following command:

```bash
pip install git+https://github.com/TalonBvV/ImageEncoder.git
```

## Usage

Please see the **`image_encoder_tutorial.ipynb`** for a complete, step-by-step guide on how to:
1.  Install the package.
2.  Download a sample dataset.
3.  Configure and run the training.
4.  Use the trained encoder for inference.
