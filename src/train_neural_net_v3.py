import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Load data
X = np.load("model_data/X.npy")
y = np.load("model_data/y.npy").reshape(-1, 1)
print("✅ Loaded X:", X.shape, ", y:", y.shape)

# Feature and target normalization
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)

y_log = np.log1p(y)  # log(1 + y) for smoother target distribution

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Neural Network with BatchNorm, LeakyReLU, Dropout
class EliteNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, optimizer
model = EliteNeuralNet(input_dim=X.shape[1])
criterion = nn.SmoothL1Loss()  # aka Huber Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Training loop
best_r2 = -np.inf
epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Evaluate after each epoch
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        preds_np = np.expm1(preds.numpy())  # inverse of log1p
        y_true = np.expm1(y_test_tensor.numpy())  # inverse of log1p

        mae = mean_absolute_error(y_true, preds_np)
        mse = mean_squared_error(y_true, preds_np)  # Calculate MSE first
        rmse = np.sqrt(mse)                         # Then take square root to get RMSE
        r2 = r2_score(y_true, preds_np)

    scheduler.step(epoch_loss)

    if r2 > best_r2:
        best_r2 = r2
        torch.save(model.state_dict(), "models/neural_net_elite_v3.pth")

    print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

print("\n✅ Best model saved to models/neural_net_elite_v3.pth")
