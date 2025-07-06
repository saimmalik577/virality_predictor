import pandas as pd
import numpy as np

# Define paths
csv_path = "data/final_processed_no_outliers/Posts_carbonbrief_18_Apr_2024_17_Apr_2025_2db6 6_CLEANED.csv"
text_embedding_path = "data/text_embeddings/Posts_carbonbrief_18_Apr_2024_17_Apr_2025_2db6 6_CLEANED_CAPTIONS.npy"
image_embedding_path = "data/image_embeddings/Posts_carbonbrief_18_Apr_2024_17_Apr_2025_2db6 6_CLEANED_img_embeddings.npy"

# Load data
df = pd.read_csv(csv_path)
text_embeds = np.load(text_embedding_path)
image_embeds = np.load(image_embedding_path)

# Basic checks
print("✅ CSV rows:", df.shape[0])
print("✅ Text embeddings:", text_embeds.shape)
print("✅ Image embeddings:", image_embeds.shape)

# Example: Print info for first post
print("\n🔍 First Post Info")
print("🖼️ Image Filename:", df['Image Filename'].iloc[0])
print("📝 Caption:", df['Caption'].iloc[0])
print("🔢 Text Embedding Shape:", text_embeds[0].shape)
print("🧠 Image Embedding Shape:", image_embeds[0].shape)
