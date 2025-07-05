import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

from generator_model import Generator
import config

# Load generator model from Hugging Face Hub
model_path = hf_hub_download(repo_id="anshumansharma0512/hawkeye-pix2pix-models", filename="gen.pth.tar")

device = config.DEVICE
gen = Generator(in_channels=3).to(device)
checkpoint = torch.load(model_path, map_location=device)
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()

# Preprocess function (no width restriction now)
def preprocess_image(img):
    img = Image.fromarray(np.array(img)).resize((256, 256))
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img

# Generate map function
def generate_map(image):
    if image is None:
        return None
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        fake = gen(input_tensor)
        fake = (fake + 1) / 2  # Rescale to [0, 1]
        fake = fake.squeeze().permute(1, 2, 0).cpu().numpy()
        fake = (fake * 255).astype(np.uint8)
    return fake

# Streamlit UI
st.title("HawkEye Vigilance Mapper 🛰️")

st.markdown("### Upload an Image or Select a Sample")

# Two-column layout
col1, col2 = st.columns(2)

# Initialize session state to store selected image
if 'selected_image' not in st.session_state:
    st.session_state['selected_image'] = None

# Upload Image Box
with col1:
    uploaded_file = st.file_uploader("Upload a Surveillance Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state['selected_image'] = Image.open(uploaded_file).convert("RGB")
    if st.session_state['selected_image'] is not None:
        st.image(st.session_state['selected_image'], caption="Uploaded Image", use_container_width=True)

# Processed Image Box
with col2:
    if st.session_state['selected_image'] is not None:
        st.image("https://via.placeholder.com/256x256?text=Processed+Image", caption="Processed Image", use_container_width=True)
    else:
        st.image("https://via.placeholder.com/256x256?text=Processed+Image", caption="Processed Image", use_container_width=True)

# Mapify Button
if st.button("Mapify"):
    if st.session_state['selected_image'] is not None:
        result = generate_map(st.session_state['selected_image'])
        if result is not None:
            with col2:
                st.image(result, caption="Processed Image", use_container_width=True)

# Sample Images Section
st.markdown("---")
st.header("Sample Images")
sample_dir = "data/val"
sample_files = sorted(os.listdir(sample_dir))[:40]

sample_cols = st.columns(4)

for idx, file in enumerate(sample_files):
    with sample_cols[idx % 4]:
        img_path = os.path.join(sample_dir, file)
        img = Image.open(img_path)
        satellite_crop = img.crop((0, 0, img.width // 2, img.height))
        if st.button(f"Sample {idx+1}", key=f"sample_{idx+1}"):
            st.session_state['selected_image'] = satellite_crop
        st.image(satellite_crop, use_container_width=True)
