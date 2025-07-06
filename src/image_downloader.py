import os
import pandas as pd
import requests
from urllib.parse import urlparse

# Define directories
CLEANED_DIR = "data/cleaned"
IMAGE_DIR = "data/images"

# Make sure the images directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Function to download an image from a URL and save it with a unique filename
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises error if response code is not 200
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

# Loop through each cleaned CSV
for filename in os.listdir(CLEANED_DIR):
    if filename.endswith(".csv"):
        print(f"\n📂 Processing file: {filename}")
        csv_path = os.path.join(CLEANED_DIR, filename)
        df = pd.read_csv(csv_path)

        image_filenames = []

        for i, row in df.iterrows():
            url = row["Image Link"]
            extension = ".jpeg"  # You can check content-type in the request header if unsure
            image_name = f"{filename.replace('.csv','')}_{i}{extension}"
            image_path = os.path.join(IMAGE_DIR, image_name)

            success = download_image(url, image_path)
            image_filenames.append(image_name if success else None)

        # Add the image filename column
        df["Image Filename"] = image_filenames

        # Save updated CSV back to cleaned folder
        df.to_csv(csv_path, index=False)
        print(f"✅ Updated CSV saved with image filenames: {filename}")
