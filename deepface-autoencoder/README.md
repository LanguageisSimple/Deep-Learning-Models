
# CelebA Autoencoder

## Overview
This project implements a **Convolutional Autoencoder** trained on the [*CelebA dataset*](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) for image compression and reconstruction.

## Features
- Learns compressed latent representation of face images
- Reconstructs images from compressed representation
- Evaluated using PSNR and SSIM metrics

## Dataset
- [CelebA Dataset (Kaggle)](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## Model
- Encoder: Convolutional layers with downsampling
- Decoder: Transposed convolution layers for reconstruction

## Results
- Average PSNR: 31.126471439997356
- Average SSIM: 0.9329655667146047

# Notes
- Model performs lossy compression
- Some blurring is expected due to reconstruction loss

# Please Fell Free to Use this Project in what ever way you like.