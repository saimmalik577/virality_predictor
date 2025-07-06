import os
import numpy as np
import pandas as pd

# --- Paths ---
csv_dir = "data/final_processed_no_outliers"
text_embed_dir = "data/text_embeddings"
image_embed_dir = "data/image_embeddings"
output_dir = "model_data"
os.makedirs(output_dir, exist_ok=True)

X = []
y = []

# --- Loop through all CSVs ---
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        print(f"📂 Processing: {filename}")
        
        csv_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(csv_path)

        text_npy_path = os.path.join(text_embed_dir, filename.replace(".csv", "_CAPTIONS.npy"))
        image_npy_path = os.path.join(image_embed_dir, filename.replace(".csv", "_img_embeddings.npy"))

        # --- Load embeddings ---
        text_embeddings = np.load(text_npy_path)
        image_embeddings = np.load(image_npy_path)

        # --- Check dimensions match ---
        if len(df) != len(text_embeddings) or len(df) != len(image_embeddings):
            print(f"❌ Skipping {filename} due to mismatch in rows and embeddings.")
            continue

        # --- Combine BERT + ResNet50 embeddings ---
        combined = np.concatenate([text_embeddings, image_embeddings], axis=1)  # shape: (num_posts, 768+2048)

        # --- Store data ---
        X.append(combined)
        y.append(df["Eng. Rate by Followers"].values)

# --- Final arrays ---
X = np.vstack(X)  # shape: (total_posts, 2816)
y = np.concatenate(y)  # shape: (total_posts,)

# --- Save final model inputs ---
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

print(f"\n✅ Final shapes -> X: {X.shape}, y: {y.shape}")
print("🎉 Model-ready data saved in 'model_data/' folder.")
