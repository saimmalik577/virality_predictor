import os
import pandas as pd
import re

# --- Define directories ---
FINAL_PROCESSED_DIR = "data/final_processed_no_outliers"
CLEANED_CAPTIONS_DIR = "data/cleaned_captions"

# Create output directory if it doesn't exist
os.makedirs(CLEANED_CAPTIONS_DIR, exist_ok=True)

# --- Text cleaning function ---
def clean_caption(text):
    text = str(text).lower()                              # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)            # remove URLs
    text = re.sub(r"[^\w\s#@]", "", text)                 # remove punctuation except # and @
    text = re.sub(r"\s+", " ", text).strip()              # trim extra whitespace
    return text

# --- Process each CSV ---
for filename in os.listdir(FINAL_PROCESSED_DIR):
    if filename.endswith(".csv"):
        print(f"🧽 Cleaning captions in: {filename}")
        
        file_path = os.path.join(FINAL_PROCESSED_DIR, filename)
        df = pd.read_csv(file_path)

        # Clean captions
        df["Cleaned Caption"] = df["Caption"].apply(clean_caption)

        # Select only the columns we need for BERT
        output_df = df[["Image Filename", "Caption", "Cleaned Caption"]]

        # Save to new folder
        output_path = os.path.join(CLEANED_CAPTIONS_DIR, filename.replace(".csv", "_CAPTIONS.csv"))
        output_df.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned captions to: {output_path}\n")
