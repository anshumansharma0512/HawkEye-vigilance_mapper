import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

from generator_model import Generator
import config

# Download the model directly from Hugging Face Hub
model_path = hf_hub_download(repo_id="anshumansharma0512/hawkeye-pix2pix-models", filename="gen.pth.tar")

# Load generator model
device = config.DEVICE
gen = Generator(in_channels=3).to(device)
checkpoint = torch.load(model_path, map_location=device)
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()


def preprocess_sample_image(img):
    img = Image.fromarray(np.array(img)).resize((256, 256))
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# Preprocess function
def preprocess_satellite_image(img):
    img = np.array(img)
    if img.shape[1] < 1200:
        st.error("Image width should be at least 1200 pixels (satellite + map).")
        return None
    img = img[:, :600, :]  # Crop left half
    img = Image.fromarray(img).resize((256, 256))
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# Generate map function
def generate_map(image, is_sample=False):
    if is_sample:
        input_tensor = preprocess_sample_image(image)
    else:
        input_tensor = preprocess_satellite_image(image)

    if input_tensor is None:
        return None

    with torch.no_grad():
        fake = gen(input_tensor)
        fake = (fake + 1) / 2  # Rescale to [0, 1]
        fake = fake.squeeze().permute(1, 2, 0).cpu().numpy()
        fake = (fake * 255).astype(np.uint8)
    return fake


# Streamlit UI
st.title("HawkEye Vigilance Mapper 🛰️")

uploaded_file = st.file_uploader("Upload a Surveillance Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Surveillance Image", use_container_width=True)

    result = generate_map(image)
    if result is not None:
        st.image(result, caption="Generated Map", use_container_width=True)

st.markdown("---")
st.header("Or Try a Sample Image")

sample_dir = "data/val"
sample_files = sorted(os.listdir(sample_dir))[:40]

cols = st.columns(4)
for idx, file in enumerate(sample_files):
    with cols[idx % 4]:
        img_path = os.path.join(sample_dir, file)
        img = Image.open(img_path)
        satellite_crop = img.crop((0, 0, img.width // 2, img.height))
        if st.button(f"Sample {idx+1}"):
            st.image(satellite_crop, caption=f"Sample {idx+1}", use_container_width=True)
            result = generate_map(satellite_crop, is_sample=True)
            if result is not None:
                st.image(result, caption="Generated Map", use_container_width=True)
