import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# Define your dataset class
class createAugment(Dataset):
    def __init__(self, X, y, dim=(32, 32), n_channels=3, transform=None):
        self.X = X
        self.y = y
        self.dim = dim
        self.n_channels = n_channels
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img_copy = img.copy()
        masked_image = self.__createMask(img_copy)

        if self.transform:
            masked_image = self.transform(masked_image)
            label = self.transform(label)

        return masked_image, label

    def __createMask(self, img):
        mask = np.full((32, 32, 3), 255, np.uint8)
        for _ in range(np.random.randint(1, 10)):
            x1, x2 = np.random.randint(1, 32), np.random.randint(1, 32)
            y1, y2 = np.random.randint(1, 32), np.random.randint(1, 32)
            thickness = np.random.randint(1, 3)
            cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

# Define your UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.contracting_block(3, 32)
        self.encoder2 = self.contracting_block(32, 64)
        self.encoder3 = self.contracting_block(64, 128)
        self.encoder4 = self.contracting_block(128, 256)

        self.upconv5 = self.expansive_block(256, 128)
        self.upconv6 = self.expansive_block(128, 64)
        self.upconv7 = self.expansive_block(64, 32)
        self.upconv8 = self.expansive_block(32, 16)

        self.final_layer = nn.Conv2d(16, 3, kernel_size=1)

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expansive_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        )
        return block

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec5 = self.upconv5(enc4)
        dec6 = self.upconv6(dec5 + enc3)
        dec7 = self.upconv7(dec6 + enc2)
        dec8 = self.upconv8(dec7 + enc1)

        final = self.final_layer(dec8)
        return final

# Function to load and split data
def load_and_split_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Store data in session state
    if 'train_dataset' not in st.session_state:
        st.session_state.train_dataset = train_dataset
        st.session_state.test_dataset = test_dataset

# Function to display sample images
def display_sample_images():
    sample_images = st.session_state.train_dataset.data[:32]
    sample_labels = st.session_state.train_dataset.targets[:32]

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(16., 8.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 8),  # creates 4x8 grid of axes
                     axes_pad=0.3,  # pad between axes in inches
                     )

    for ax, image, label in zip(grid, sample_images, sample_labels):
        ax.imshow(image)
        ax.set_title(class_names[label])
        ax.axis('off')  # Hide axes for better visualization

    # Display the plot in Streamlit
    st.pyplot(fig)

# Function to build the UNet model
def build_unet_model():
    device = st.session_state.device
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = st.session_state.train_loader

    # Training loop
    num_epochs = 6
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for masked_data, original_data in train_loader:
            masked_data, original_data = masked_data.to(device), original_data.to(device)
            optimizer.zero_grad()
            output = model(masked_data)
            loss = criterion(output, original_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

    st.session_state.model = model
    torch.save(model.state_dict(), 'unet_inpainting.pth')

# Function to visualize inpainting
def visualize_inpainting(num_images=5):
    device = st.session_state.device
    model = st.session_state.model
    test_dataset = st.session_state.test_dataset

    model.eval()
    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))

    for i in range(num_images):
        masked_img, original_img = test_dataset[i]

        with torch.no_grad():
            masked_img = masked_img.unsqueeze(0).to(device)
            output = model(masked_img)
            output = output.squeeze(0).cpu()

        axes[i, 0].imshow(original_img.permute(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masked_img.squeeze(0).permute(1, 2, 0).cpu())
        axes[i, 1].set_title('Masked Image')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(output.permute(1, 2, 0))
        axes[i, 2].set_title('Reconstructed Image')
        axes[i, 2].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# Function to inpaint custom images
def inpaint_custom_image(uploaded_file, target_size=(32, 32)):
    device = st.session_state.device
    model = st.session_state.model

    custom_masked_image = Image.open(uploaded_file)
    custom_masked_image = custom_masked_image.resize(target_size)
    custom_masked_image = np.array(custom_masked_image) / 255.0
    custom_masked_image = transforms.ToTensor()(custom_masked_image).unsqueeze(0).float().to(device)  # Ensure float type

    model.eval()
    with torch.no_grad():
        inpainted_image = model(custom_masked_image).cpu().squeeze(0).permute(1, 2, 0).numpy()

    inpainted_image = (inpainted_image * 255).astype(np.uint8)

    return inpainted_image
