# Image Colorization using Deep Learning

This project implements a colorization model for grayscale images based on the approach presented in the paper *"Colorful Image Colorization"* by Zhang, Isola, and Efros (ECCV 2016). The model leverages a deep learning framework to learn how to predict the color information of an image based on its grayscale input, producing vibrant colorized versions of black-and-white photographs.

## Overview

This repository implements an end-to-end pipeline for training and testing a model to colorize grayscale images. The model is based on the architecture and ideas from *"Colorful Image Colorization"* by Zhang et al., which uses a convolutional neural network (CNN) to predict the color (ab channels in the LAB color space) of a grayscale input (L channel). The network is trained to map grayscale images to their full-color counterparts.

Key features:
- Converts images from RGB to LAB color space (L channel as input, ab channels as targets)
- Utilizes a deep convolutional network to predict color values from grayscale
- Includes training, evaluation, and visualization components
- Easily customizable for training with custom datasets

## Installation

To run the project, you will need a Python environment with the following dependencies:

```bash
pip install torch torchvision scikit-image matplotlib tqdm scikit-learn Pillow torchinfo
```

Additionally, you'll need a dataset of color images for training. The dataset should consist of RGB images that will be preprocessed into LAB color space (grayscale L channel as input, color a and b channels as targets).

## Usage

1. **Prepare your dataset**: Ensure you have a folder of color images.
2. **Preprocessing**: The images are processed into the LAB color space, with the L channel as input and the a and b channels as targets.
3. **Training**: You can train the model by running the `train_model()` function.
4. **Testing**: After training, use the `test_model()` function to evaluate the performance on a test set.
5. **Visualization**: Use `visualize_colorization()` to display and compare the colorized results.

### Pre-trained Model

You can load a pre-trained model using the following function:

```python3
def colorization_model(pretrained=True):
    model = ColorfulGenerator()
    if pretrained:
        # Load pre-trained weights from a URL
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth', map_location='cpu', check_hash=True))
    return model
```

### Training the Model

The `train_model()` function handles the training of the model, allowing you to specify the model architecture, criterion (loss function), optimizer, and learning rate scheduler. Here is how you can train the model:

```python3
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # Training loop implementation
```

### Testing the Model

Once the model is trained, you can evaluate it on a test dataset using the `test_model()` function, which computes the loss and provides a performance summary:

```python3
def test_model(model, criterion, test_loader):
    # Test loop implementation
```

### Visualizing Colorization Results

The `visualize_colorization()` function displays colorized images alongside their corresponding grayscale inputs and original color images, which helps you evaluate the quality of the colorization:

```python3
def visualize_colorization(model, test_loader, num_images=5):
    # Visualization implementation
```
