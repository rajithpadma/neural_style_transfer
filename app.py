import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Preprocessing ---
def preprocess_image(image, target_height, target_width):
    """
    Preprocess the image: resize, crop, and normalize.
    """
    transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def tensor_to_image(tensor):
    """
    Converts a tensor to a PIL Image.
    """
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor.mul(torch.tensor([0.229, 0.224, 0.225], device=device)).add(
        torch.tensor([0.485, 0.456, 0.406], device=device)
    )
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# --- Style Transfer Function ---
def perform_style_transfer(content_tensor, style_tensor, num_steps=300, style_weight=1e6, content_weight=1):
    """
    Performs style transfer using PyTorch.
    """
    model = models.vgg19(pretrained=True).features.to(device).eval()

    # Extract features
    def extract_features(image):
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1"
        }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    # Compute content and style loss
    content_features = extract_features(content_tensor)
    style_features = extract_features(style_tensor)
    target = content_tensor.clone().requires_grad_(True).to(device)

    style_grams = {layer: torch.matmul(style_features[layer].squeeze(0), style_features[layer].squeeze(0).T) 
                   for layer in style_features}

    optimizer = torch.optim.Adam([target], lr=0.003)

    for step in range(num_steps):
        target_features = extract_features(target)

        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

        style_loss = 0
        for layer in style_features:
            target_gram = torch.matmul(target_features[layer].squeeze(0), target_features[layer].squeeze(0).T)
            style_loss += torch.mean((target_gram - style_grams[layer]) ** 2)

        loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return target

# --- Streamlit Interface ---
st.title("Style Transfer with PyTorch")
st.write("Upload a content image and a style image to generate a stylized result.")

# Upload content and style images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

# Parameters for image preprocessing
IMG_WIDTH = 400
IMG_HEIGHT = 300

# Show uploaded images
if content_file:
    content_image = Image.open(content_file).convert("RGB")
    st.image(content_image, caption="Content Image", use_column_width=True)

if style_file:
    style_image = Image.open(style_file).convert("RGB")
    st.image(style_image, caption="Style Image", use_column_width=True)

# Perform style transfer
if content_file and style_file and st.button("Generate Stylized Image"):
    st.write("Performing style transfer. Please wait...")

    # Preprocess images
    content_tensor = preprocess_image(content_image, IMG_HEIGHT, IMG_WIDTH)
    style_tensor = preprocess_image(style_image, IMG_HEIGHT, IMG_WIDTH)

    # Perform style transfer
    try:
        stylized_tensor = perform_style_transfer(content_tensor, style_tensor)
        stylized_image = tensor_to_image(stylized_tensor)

        # Display the stylized image
        st.image(stylized_image, caption="Stylized Image", use_column_width=True)
        st.success("Style transfer completed successfully!")

        # Option to download the result
        if st.button("Download Stylized Image"):
            output_path = "stylized_result.png"
            stylized_image.save(output_path)
            with open(output_path, "rb") as file:
                st.download_button("Download Image", file, file_name="stylized_image.png")
    except Exception as e:
        st.error(f"Error during style transfer: {e}")

st.write("---")
st.write("Adjust `IMG_WIDTH` and `IMG_HEIGHT` in the code for resolution tuning.")
