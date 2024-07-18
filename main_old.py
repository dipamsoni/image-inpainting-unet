import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import streamlit as st
from modules import image_inpainting_old

# Streamlit app configuration
st.set_page_config(layout="wide")
# Streamlit app
st.title("Image InPainting Tool")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate to a page", ["Check GPU", "Load and Split data", "Display Sample Images", "Build Model", "Display Masked Images", "Custom Inpainting"])

if page == "Check GPU":
    st.header("Check  GPU availability")
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device:
        st.write("GPU name:" + torch.cuda.get_device_name(0))
        st.success(f"GPU available")
        st.session_state.device = device
    else:
        st.session_state.device = device
        st.error(f"GPU not available")

elif page == "Load and Split data":
    st.header("Load and split the data")

    # Button to add item to session state
    if st.button("Add Item"):
        image_inpainting_old.add_data()
        st.success(f"Data Split completed")

elif page == "Display Sample Images":
    st.header("Display Sample Images")
    image_inpainting_old.display_sample_images()

elif page == "Build Model":
    st.header("Build UNET Model")
    # Button to add item to session state
    if st.button("Build Model"):
        image_inpainting_old.unet_model()
        st.success("Model saved in session")

elif page == "Display Masked Images":
    st.header("Masked Images")
    # Visualize the inpainting results
    image_inpainting_old.visualize_inpainting()

elif page == "Custom Inpainting":
    st.header("Custom image to do InPainting")
    custom_masked_image_path = 'data/images/masked_Img1.jpg'
    image_inpainting_old.inpaint_custom_image(custom_masked_image_path)

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
