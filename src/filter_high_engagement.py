import os
import pandas as pd

# 📁 Define folders
PROCESSED_DIR = "data/processed"
FINAL_DIR = "data/final_processed"
os.makedirs(FINAL_DIR, exist_ok=True)

# ⚙️ Filtering threshold
ENGAGEMENT_THRESHOLD = 15.0

# 🔁 Go through all CSVs in processed folder
for filename in os.listdir(PROCESSED_DIR):
    if filename.endswith(".csv"):
        print(f"📂 Processing {filename}...")

        file_path = os.path.join(PROCESSED_DIR, filename)

        try:
            df = pd.read_csv(file_path)

            # 📉 Filter out rows where Engagement Rate > 15
            df_filtered = df[df["Eng. Rate by Followers"] <= ENGAGEMENT_THRESHOLD]

            # 💾 Save to final processed directory
            output_path = os.path.join(FINAL_DIR, filename)
            df_filtered.to_csv(output_path, index=False)

            print(f"✅ Saved filtered file to {output_path}\n")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
