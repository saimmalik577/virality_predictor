import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# 📁 Define paths
FINAL_CAPTION_DIR = "data/cleaned_captions"
EMBEDDING_DIR = "data/embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# 📦 Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# 🧠 Function to compute BERT embedding for a single caption
def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# 🔁 Loop through all caption CSVs
for filename in os.listdir(FINAL_CAPTION_DIR):
    if filename.endswith(".csv"):
        print(f"\n📂 Processing {filename}")
        csv_path = os.path.join(FINAL_CAPTION_DIR, filename)
        df = pd.read_csv(csv_path)

        embeddings = []
        valid_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding: {filename}"):
            caption = row["Cleaned Caption"]
            try:
                emb = get_bert_embedding(caption)
                embeddings.append(emb)
                valid_rows.append(row)
            except Exception as e:
                print(f"❌ Failed to embed caption: {e}")

        # 💾 Save embeddings and corresponding filtered CSV
        if embeddings:
            embeddings_array = np.array(embeddings)
            np.save(os.path.join(EMBEDDING_DIR, filename.replace(".csv", ".npy")), embeddings_array)
            pd.DataFrame(valid_rows).to_csv(os.path.join(EMBEDDING_DIR, filename), index=False)
            print(f"✅ Saved embeddings + valid rows for {filename}")
        else:
            print(f"⚠️ No embeddings saved for {filename}")
