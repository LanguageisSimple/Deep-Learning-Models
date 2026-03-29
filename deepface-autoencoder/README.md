
# CelebA Autoencoder

## Overview
This project implements a Convolutional Autoencoder trained on the CelebA dataset for image compression and reconstruction.

## Features
- Learns compressed latent representation of face images
- Reconstructs images from compressed representation
- Evaluated using PSNR and SSIM metrics

## Dataset
- CelebA Dataset (Kaggle)

## Model
- Encoder: Convolutional layers with downsampling
- Decoder: Transposed convolution layers for reconstruction

## Results
- PSNR: ~31 dB
- SSIM: ~0.93

## Usage

### Run Inference
```bash
python inference.py path_to_image.jpg

Notes
Model performs lossy compression
Some blurring is expected due to reconstruction loss
Author

Autoencoder project for Deep Learning experiment
