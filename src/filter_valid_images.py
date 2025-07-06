import os
import pandas as pd

CLEANED_DIR = "data/cleaned"
PROCESSED_DIR = "data/processed"

# Make sure the output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

for filename in os.listdir(CLEANED_DIR):
    if filename.endswith(".csv"):
        print(f"📂 Processing {filename}...")

        filepath = os.path.join(CLEANED_DIR, filename)
        df = pd.read_csv(filepath)

        # Remove rows where the image couldn't be downloaded
        df_filtered = df[df["Image Filename"].notna()]

        # Save to processed directory
        output_path = os.path.join(PROCESSED_DIR, filename)
        df_filtered.to_csv(output_path, index=False)

        print(f"✅ Saved to {output_path} ({len(df_filtered)} rows)\n")
