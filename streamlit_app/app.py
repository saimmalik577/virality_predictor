import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Set environment variable to fix torch classes watchdog issue
os.environ["STREAMLIT_WATCHDOG_INSENSITIVE_TRIGGER"] = "true"

# Import ResNet50_Weights correctly
from torchvision.models import ResNet50_Weights

# Configure page
st.set_page_config(
    page_title="Social Media Virality Predictor",
    page_icon="📊",
    layout="wide"
)

# Define the neural network architecture (same as training)
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Function to load neural network model
@st.cache_resource
def load_prediction_model():
    # Load X to get input dimensions
    try:
        X = np.load('../model_data/X.npy')
        input_dim = X.shape[1]
    except:
        # Default input dim: BERT (768) + ResNet50 (2048)
        input_dim = 2816
    
    model = NeuralNetwork(input_dim=input_dim)
    model_path = '../models/neural_net_model.pth'
  
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model, input_dim
    else:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, input_dim

# Function to load ResNet-50 model (excluding classification head)
@st.cache_resource
def load_resnet_model():
    # Fixed: correct spelling of resnet50 and ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove the classification layer to get embeddings (just like in your code)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

# Function to load BERT model
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

# Function to get image embeddings (following your implementation)
def get_image_embedding(image, model):
    # Same transformations as in your code
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        embedding = model(image_tensor)
        # Match your implementation
        embedding = embedding.squeeze().numpy()
    
    return embedding

# Function to get text embeddings (following your implementation)
def get_bert_embedding(text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# Function to combine embeddings (following your implementation)
def combine_embeddings(text_embedding, image_embedding):
    # Concatenate the embeddings exactly as in your data preparation code
    combined = np.concatenate([text_embedding, image_embedding])
    return combined

# Function to make a prediction
def make_prediction(model, features):
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        prediction = model(features_tensor)
        return prediction.item()

# Main app function
def main():
    st.title("📊 Social Media Post Virality Predictor")
    
    # Load models with progress indicators
    with st.spinner("Loading prediction model..."):
        prediction_model, input_dim = load_prediction_model()
    
    with st.spinner("Loading embedding models..."):
        resnet_model = load_resnet_model()
        bert_tokenizer, bert_model = load_bert_model()
    
    if prediction_model is None:
        st.error("Failed to load the prediction model. Please check if the model file exists.")
        return
    
    # Sidebar for navigation
    page = st.sidebar.radio("Navigation", ["Predict Post Virality", "View Model Architecture", "Model Performance"])
    
    # Page: Predict Post Virality
    if page == "Predict Post Virality":
        st.header("Predict Engagement Rate")
        st.write("Upload an image and enter a caption to predict the virality of your social media post.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Upload image
            uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            # Text caption input
            caption = st.text_area("Enter post caption", height=150)
            
            # Predict button
            if st.button("Predict Engagement Rate", type="primary", disabled=(uploaded_file is None or not caption)):
                if uploaded_file is None:
                    st.warning("Please upload an image.")
                elif not caption:
                    st.warning("Please enter a caption.")
                else:
                    with st.spinner("Processing image and text..."):
                        # Get image embedding
                        image_embedding = get_image_embedding(image, resnet_model)
                        
                        # Get text embedding
                        text_embedding = get_bert_embedding(caption, bert_tokenizer, bert_model)
                        
                        # Combine embeddings
                        combined_embedding = combine_embeddings(text_embedding, image_embedding)
                        
                        # Verify dimensions
                        expected_dim = input_dim
                        actual_dim = combined_embedding.shape[0]
                        
                        if actual_dim != expected_dim:
                            st.error(f"Dimension mismatch: Expected {expected_dim}, got {actual_dim}.")
                            st.info("This could be because your model was trained with different embedding models.")
                        else:
                            # Make prediction
                            prediction = make_prediction(prediction_model, combined_embedding)
                            
                            # Display prediction with proper formatting
                            engagement_percentage = prediction 
                            
                            st.success(f"Predicted Engagement Rate: {engagement_percentage:.2f}%")
                            
                            # Create visualization
                            fig, ax = plt.subplots(figsize=(10, 4))

                            # Create a horizontal bar chart
                            ax.barh(["Engagement Rate"], [engagement_percentage], color='#1f77b4')

                            # Set limits: Axis always max at 15%
                            ax.set_xlim(0, 15)
                            ax.set_xlabel('Engagement Rate (%)')

                            # Add the percentage in the bar (make sure text fits!)
                            bar_value = min(engagement_percentage, 15)  # Clip in case model overpredicts
                            ax.text(bar_value + 0.4, 0, f"{engagement_percentage:.2f}%", 
                                va='center', fontweight='bold')

                            ax.set_yticks([])  # Remove y axis label
                            ax.grid(axis='x', linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig)
                            
                            # Add engagement rate interpretation
                            st.subheader("Engagement Rate Interpretation")
                            
                            if engagement_percentage < 1.0:
                                st.warning("📉 Below Average Engagement: This post may underperform.")
                                st.write("Consider reviewing your content strategy.")
                            elif engagement_percentage < 3.0:
                                st.info("📊 Average Engagement: This post will likely perform normally.")
                                st.write("Minor optimizations could improve performance.")
                            elif engagement_percentage < 6.0:
                                st.success("📈 Good Engagement: This post should perform well!")
                                st.write("This content is resonating with your audience.")
                            else:
                                st.success("🔥 Viral Potential: This post could go viral!")
                                st.write("This content has very strong engagement potential.")
                                
                            # Add recommendations based on the prediction
                            st.subheader("Recommendations to Improve Engagement")
                            st.write("- **Best Time to Post**: Consider posting during peak audience activity times")
                            st.write("- **Hashtags**: Include 5-10 relevant hashtags to increase discoverability")
                            st.write("- **Call to Action**: Ask a question or encourage interaction")
                            st.write("- **Engage Early**: Respond to comments within the first hour of posting")
    
    # Page: View Model Architecture
    elif page == "View Model Architecture":
        st.header("Model Architecture & Data Pipeline")

        # Display with tabs for different components
        tab1, tab2, tab3 = st.tabs(["Data Pipeline", "Model Architecture", "Input Features"])
        
        with tab1:
            st.subheader("Data Pipeline")
            st.write("""
            This model uses a two-branch architecture to process images and text separately before combining them:
            
            1. **Image Processing**: ResNet-50 CNN extracts 2048-dimensional visual features
            2. **Text Processing**: BERT extracts 768-dimensional semantic features from captions
            3. **Feature Fusion**: Concatenation of image and text embeddings
            4. **Prediction**: Neural network trained to predict engagement rate
            """)
            
            # Add a diagram
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*D6J1XZGC3wLdJnPKx8nNeA.jpeg", 
                    caption="Multimodal architecture combining visual and textual features", 
                    use_column_width=True)
        
        with tab2:
            st.subheader("Neural Network Architecture")
            st.code("""
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # First dense layer
            nn.ReLU(),                  # Activation function
            nn.Dropout(0.3),            # Regularization to prevent overfitting
            nn.Linear(512, 128),        # Second dense layer
            nn.ReLU(),                  # Activation function
            nn.Linear(128, 1)           # Output layer (engagement prediction)
        )

    def forward(self, x):
        return self.model(x)
            """)
            
            st.write(f"- **Input size**: {input_dim} features (combined BERT + ResNet embeddings)")
            st.write("- **Hidden layers**: 512 → 128 neurons with ReLU activation")
            st.write("- **Regularization**: Dropout (30%) to prevent overfitting")
            st.write("- **Output**: Single neuron predicting engagement rate")
            
        with tab3:
            st.subheader("Input Feature Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Image Features (ResNet-50)**")
                st.write("- Dimensions: 2048")
                st.write("- Preprocessing: Resize to 224×224, normalize")
                st.write("- Source: Penultimate layer of ResNet-50")
                st.write("- Captures: Visual patterns, objects, composition")
            
            with col2:
                st.write("**Text Features (BERT)**")
                st.write("- Dimensions: 768")
                st.write("- Preprocessing: Tokenization, truncation (max 512 tokens)")
                st.write("- Source: [CLS] token embedding from BERT")
                st.write("- Captures: Semantic meaning, tone, hashtags")
    
    # Page: Model Performance
    elif page == "Model Performance":
        st.header("Model Performance")
        
        try:
            # Load data
            X = np.load("../model_data/X.npy")
            y = np.load("../model_data/y.npy")
            
            # Display basic stats
            st.write(f"Dataset size: {len(y)} posts")
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            
            # Make predictions on all data
            prediction_model.eval()
            with torch.no_grad():
                predictions = prediction_model(X_tensor).numpy()
            
            # Calculate metrics
            mae = mean_absolute_error(y, predictions)
            rmse = mean_squared_error(y, predictions) ** 0.5
            r2 = r2_score(y, predictions)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
                st.write("Average magnitude of prediction errors")
                
            with col2:
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                st.write("Root of average squared prediction errors")
                
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
                st.write("Proportion of variance explained by the model")
            
            # Create tabs for different visualizations
            vis_tab1, vis_tab2, vis_tab3 = st.tabs(["Actual vs Predicted", "Error Distribution", "Residual Plot"])
            
            with vis_tab1:
                st.subheader("Actual vs Predicted Engagement Rates")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot scatter with alpha for density visualization
                ax.scatter(y, predictions, alpha=0.5, color='#1f77b4')
                
                # Add perfect prediction line
                max_val = max(np.max(y), np.max(predictions))
                min_val = min(np.min(y), np.min(predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect predictions')
                
                ax.set_xlabel("Actual Engagement Rate")
                ax.set_ylabel("Predicted Engagement Rate")
                ax.set_title("Model Predictions vs Actual Values")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                st.write("This plot shows how well the model predictions match actual engagement rates. Points on the red line would be perfect predictions.")
            
            with vis_tab2:
                st.subheader("Error Distribution")
                
                errors = (y.flatten() - predictions.flatten()) 
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.hist(errors, bins=30, alpha=0.7, color='#2ca02c')
                ax2.axvline(x=0, color='r', linestyle='--', label='Zero Error')
                ax2.set_xlabel("Prediction Error")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Distribution of Prediction Errors")
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                st.pyplot(fig2)
                st.write("This histogram shows the distribution of prediction errors. Ideally, errors should be centered around zero and follow a normal distribution.")
            
            with vis_tab3:
                st.subheader("Residual Plot")
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.scatter(predictions.flatten(), y.flatten() - predictions.flatten(), alpha=0.5, color='#ff7f0e')
                ax3.axhline(y=0, color='r', linestyle='--')
                ax3.set_xlabel("Predicted Values")
                ax3.set_ylabel("Residuals (Actual - Predicted)")
                ax3.set_title("Residual Plot")
                ax3.grid(True, alpha=0.3)
                
                st.pyplot(fig3)
                st.write("This plot shows residuals against predicted values. Ideally, residuals should be randomly distributed around zero with no clear pattern.")
                
        except Exception as e:
            st.error(f"Error loading or processing evaluation data: {e}")
            st.write("Make sure your model data files exist at 'model_data/X.npy' and 'model_data/y.npy'")
            
            # Display dummy metrics for demonstration
            st.subheader("Sample Metrics (Demo Only)")
            st.warning("Using placeholder data since actual model data couldn't be loaded")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", "0.0123")
            col2.metric("RMSE", "0.0189")
            col3.metric("R²", "0.723")

if __name__ == "__main__":
    main()