import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Define paths
CSV_DIR = "data/final_processed_no_outliers"
IMG_DIR = "data/images"
EMBED_DIR = "data/image_embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

# Load pretrained ResNet50 model (excluding classification head)
model = resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Preprocessing for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

# Loop over CSV files
for file in os.listdir(CSV_DIR):
    if file.endswith(".csv"):
        print(f"\n📂 Processing {file}...")

        # Read the CSV
        df = pd.read_csv(os.path.join(CSV_DIR, file))

        # Prepare a list to hold embeddings
        image_embeddings = []

        for img_name in tqdm(df["Image Filename"], desc="🔍 Extracting"):
            try:
                img_path = os.path.join(IMG_DIR, img_name)
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    embedding = model(image).squeeze().numpy()  # (2048,)

                image_embeddings.append(embedding)
            except Exception as e:
                print(f"❌ Failed to process {img_name}: {e}")
                image_embeddings.append(np.zeros(2048))  # fallback

        # Save all embeddings to .npy file
        embed_name = file.replace(".csv", "_img_embeddings.npy")
        np.save(os.path.join(EMBED_DIR, embed_name), np.array(image_embeddings))
        print(f"✅ Saved to {embed_name}")
