# Social Media Virality Predictor

This repository contains the code, models, and data pipelines for **predicting the virality of sustainability-related social media content using multimodal deep learning**. Developed as the BSc thesis project of *Saim A.D. Malik* under the supervision of **Prof. Dr. Omid Fatahi Valilai** at [Constructor University Bremen](https://constructor.university), this work combines **state-of-the-art natural language processing** and **computer vision models** to predict **engagement rate** from post content.

---

## Project Overview

With growing concerns over climate change and sustainability, platforms like Instagram have become central to awareness campaigns. However, the success of such posts often depends on subjective content quality. This project builds a tool that can **predict engagement rate** (likes/comments relative to followers) based solely on a post's **image and caption**.

### Key Components:

* **BERT (text):** Extracts semantic features from captions
* **ResNet-50 (image):** Extracts visual features from post images
* **Feedforward Neural Network:** Learns to predict engagement rate from combined features
* **Streamlit Web App:** Allows users to upload content and receive engagement predictions instantly

---

## Repository Structure

```plaintext
├── streamlit_app/
│   └── app.py                # Main app interface
├── model_data/
│   ├── X.npy                 # Combined input embeddings
│   └── y.npy                 # Target engagement values
├── models/
│   ├── neural_net_model.pth  # Trained PyTorch model
│   └── xgboost_tuned.json    # Optional baseline model
├── data/
│   ├── cleaned_captions/     # Cleaned Instagram captions
│   ├── text_embeddings/      # BERT embeddings
│   ├── image_embeddings/     # ResNet-50 image embeddings
│   └── final_processed_no_outliers/  # Final dataset used for training
├── notebooks/
│   ├── 01_exploration.ipynb  # Data exploration
│   └── 02_model_dev.ipynb    # Model training and evaluation
├── src/
│   └── *.py                  # All training, embedding, preprocessing scripts
├── requirements.txt
└── README.md
```

---

## How to Run the App

1. Clone the repository

```bash
git clone https://github.com/saimmalik577/virality_predictor.git
cd virality_predictor
```

2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
cd streamlit_app
streamlit run app.py
```

5. Usage

* Upload a post image
* Paste your caption
* Click "Predict Engagement Rate"
* View the result and recommendations

---

## Model Architecture

### Embedding Strategy

* Text (BERT base): 768-dim \[CLS] embedding
* Image (ResNet-50): 2048-dim global pooled feature vector
* Combined Vector: 2816-dim input to model

### Neural Network Design

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

Trained using MSELoss and Adam optimizer.

---

## 📊 Model Performance

| Metric                  | Training Set | Validation Set | Gap   |
| ----------------------- | ------------ | -------------- | ----- |
| R² Score                | 0.625        | 0.444          | 0.181 |
| Mean Absolute Error     | 0.131        | 0.144          | 0.013 |
| Root Mean Squared Error | 0.266        | 0.325          | 0.059 |

Baseline comparison:

| Model          | R²    | MAE   | RMSE  |
| -------------- | ----- | ----- | ----- |
| XGBoost        | 0.13  | 0.21  | 0.405 |
| Random Forest  | 0.00  | 0.245 | 0.432 |
| Neural Network | 0.444 | 0.144 | 0.325 |

Evaluation is based on holdout validation from **2,700** Instagram posts from 10 verified sustainability organizations.

Visualizations available in the "Model Performance" tab of the app.

---

## Data Sources

Instagram posts scraped from:

* @greenpeace
* @wwf
* @ecowatch
* @carbonbrief
* @nature\_org
* @climatereality
* @earthdaynetwork
* @greenmatters
* @oceana
* @rainforesttrust



**Timeframe**: 18 April 2024 – 17 April 2025
**Volume**: \~3,200 posts (of which 2,787 used in final dataset)

Metadata includes: likes, comments, followers, caption, image URL

Preprocessing involved:

* Cleaning text (removing links, emojis, hashtags)
* Filtering valid images
* Outlier removal based on engagement percentiles

---

## Technologies Used

* **Streamlit** – UI framework
* **PyTorch** – Model building & inference
* **Transformers (HuggingFace)** – BERT model
* **Torchvision** – ResNet image model
* **Pandas/Numpy** – Data preprocessing
* **Matplotlib/Seaborn** – Visualization
* **XGBoost** – Baseline modeling
* **SHAP** – Model interpretability 

---

## Citation

If you use this code or model in your work, please cite:

> Malik, Saim A.D. “Predicting the Virality of Sustainability-Related Social Media Content Using Multimodal Deep Learning.” Bachelor Thesis, Constructor University Bremen, 2025. Supervised by Prof. Dr. Omid Fatahi Valilai.

---

## Contact

**Saim A.D. Malik**
[saimadmalik577@gmail.com](mailto:saimadmalik577@gmail.com)

---

## License

This repository is for academic use only. Commercial redistribution of trained models or datasets is not permitted without explicit permission from the author.
