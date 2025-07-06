import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ----- 1. Load data -----
X = np.load("model_data/X.npy")
y = np.load("model_data/y.npy")

# ----- 2. Convert to PyTorch tensors -----
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ----- 3. Split -----
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----- 4. Define model architecture (must match original) -----
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

model = NeuralNetwork(input_dim=X.shape[1])

# ----- 5. Load model params -----
model_path = "models/neural_net_model.pth"
if not os.path.exists(model_path):
    raise RuntimeError(f"Pre-trained weights not found at {model_path}!")
model.load_state_dict(torch.load(model_path))
model.eval()

# ----- 6. Evaluate -----
def eval_dataloader(loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(out.numpy())
    return np.array(y_true).flatten(), np.array(y_pred).flatten()

y_train_true, y_train_pred = eval_dataloader(train_loader)
y_val_true, y_val_pred = eval_dataloader(val_loader)

def print_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name} set:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

print_metrics(y_train_true, y_train_pred, "TRAIN")
print_metrics(y_val_true, y_val_pred, "VALIDATION")