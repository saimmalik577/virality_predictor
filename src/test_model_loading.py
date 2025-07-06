import torch

# Correct path without extra spaces
model_path = 'models/neural_net_model.pth'

# Load the model
model = torch.load(model_path)

# Check if the model is loaded correctly
print("Model loaded successfully")
