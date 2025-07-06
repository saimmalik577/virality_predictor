import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load the combined feature vectors and target values
X = np.load("model_data/X.npy")
y = np.load("model_data/y.npy")
print("✅ Loaded X:", X.shape, ", y:", y.shape)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Create dataset and dataloaders
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define a simple feedforward neural network
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

# Initialize model, loss, optimizer
model = NeuralNetwork(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🧠 Training Neural Network...")

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss / len(train_loader):.4f}")

# Evaluate
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(outputs.numpy())

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
r2 = r2_score(y_true, y_pred)

print("\n📊 Neural Network Evaluation:")
print("MAE: ", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²:  ", round(r2, 4))

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/neural_net_model.pth")
print("✅ Neural Network model saved to models/neural_net_model.pth")
