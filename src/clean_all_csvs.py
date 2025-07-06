import os
import pandas as pd
import re

# 📁 Define directories
RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

# ✅ Columns we want to keep for training and modeling
columns_to_keep = [
    "Caption",
    "Image Link",
    "Eng. Rate by Followers",
    "Date",
    "Type"
]

# 🔁 Loop through all CSVs
for filename in os.listdir(RAW_DIR):
    if filename.endswith(".csv"):
        try:
            print(f"📂 Processing {filename}...")

            file_path = os.path.join(RAW_DIR, filename)

            # 🔍 Read CSV using correct separator
            df = pd.read_csv(file_path, sep=';', encoding="utf-8")

            # 🧹 Clean column names
            df.columns = df.columns.str.strip()

            # ✅ Drop completely empty rows
            df.dropna(how="all", inplace=True)

            # ✂️ Keep only relevant columns
            df = df[columns_to_keep]

            # ✨ Clean string columns (remove non-printable/invisible characters)
            for col in ["Caption", "Type"]:
                df[col] = df[col].astype(str).apply(lambda x: re.sub(r"[^\x00-\x7F]+", " ", x).strip())

            # 📅 Convert Date column to datetime
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

            # 💡 Ensure numeric values are actually numeric
            df["Eng. Rate by Followers"] = pd.to_numeric(df["Eng. Rate by Followers"], errors='coerce')

            # 🚿 Drop rows with any crucial missing values
            df.dropna(subset=["Caption", "Image Link", "Eng. Rate by Followers", "Date", "Type"], inplace=True)

            # 💾 Save cleaned version
            output_path = os.path.join(CLEANED_DIR, filename.replace(".csv", "_CLEANED.csv"))
            df.to_csv(output_path, index=False)

            print(f"✅ Saved cleaned file to {output_path}\n")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
