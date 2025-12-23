Project Overview

This project implements Neural Style Transfer using a pre-trained MobileNetV2 deep learning model in TensorFlow. Neural Style Transfer is a computer vision technique that blends two images:

A content image (the actual subject or structure)

A style image (the artistic pattern, texture, or painting style)

The objective of this project is to generate a new image that preserves the structural features of the content image while adopting the artistic style of the style image. MobileNetV2 is used as the feature extraction backbone to ensure faster computation, efficient processing, and reduced computational cost compared to heavier architectures.

This project demonstrates the practical application of transfer learning, convolutional neural networks, and optimization-based image generation techniques in an efficient and scalable way.

Problem Statement

Traditional artistic recreation of images requires manual creativity, artistic skill, and time. Generating stylized images using deep learning eliminates this manual effort and automates the artistic rendering process. However, neural style transfer is often computationally expensive and slow when using large networks.

This project addresses the challenge by utilizing MobileNetV2, a lightweight and high-performance neural network model, to perform efficient and high-quality style transfer.

Objectives

• To implement neural style transfer using deep learning
• To generate stylized images by combining content and style features
• To use MobileNetV2 for efficient and optimized feature extraction
• To demonstrate image processing, neural networks, and optimization in a real application

Technology Stack

Programming Language: Python
Framework: TensorFlow / Keras
Model: MobileNetV2 (Pre-trained on ImageNet)
Supporting Libraries: NumPy, Matplotlib, PIL

Key Features

• Uses a lightweight MobileNetV2 model for style and content feature extraction
• Extracts different convolutional layer outputs for style and content representation
• Computes style loss using Gram Matrices
• Computes combined loss using weighted content and style contributions
• Optimizes image using gradient descent
• Produces a final stylized output image
• Displays generated image visually

How the System Works

The content image and style image are loaded and preprocessed

MobileNetV2 extracts style and content features from specified neural layers

Gram matrices are computed for style feature representation

A generated image is initialized and iteratively optimized

Loss is computed using a combination of content loss and style loss

Gradients are applied to update the generated image

The final stylized image is produced and displayed

Input Requirements

Content Image: A regular image containing the subject
Style Image: Any artistic painting, pattern, or design image

Both images must be placed in the working directory and named appropriately (example: content.jpg and style.jpg)

Output

The system produces a stylized image that retains the original structure of the content image while applying the artistic texture and color patterns of the style image. The final image is displayed using Matplotlib.

Execution Requirements

Python installed
TensorFlow installed
MobileNetV2 pre-trained model (auto downloaded by TensorFlow)
