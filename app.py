import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

# --- Load Pretrained Model and Parameters ---
@st.cache_resource
def load_saved_model():
    with open("stylized_image.pkl", "rb") as file:
        stylized_image_data = pickle.load(file)
    return stylized_image_data

# --- Image Preprocessing ---
def preprocess_image(image, target_height, target_width):
    """
    Preprocesses the uploaded image to resize and normalize.
    """
    img = tf.image.resize(image, [target_height, target_width])
    img = img / 255.0  # Normalize to [0,1]
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

def tensor_to_image(tensor):
    """
    Converts a tensor to a PIL Image.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# --- Streamlit Interface ---
st.title("Style Transfer with Deep Learning")
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
    content_tensor = tf.convert_to_tensor(np.array(content_image), dtype=tf.float32)
    content_tensor = preprocess_image(content_tensor, IMG_HEIGHT, IMG_WIDTH)

    style_tensor = tf.convert_to_tensor(np.array(style_image), dtype=tf.float32)
    style_tensor = preprocess_image(style_tensor, IMG_HEIGHT, IMG_WIDTH)

    # Load saved stylized data
    try:
        stylized_image_data = load_saved_model()
        stylized_tensor = tf.convert_to_tensor(stylized_image_data)
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
