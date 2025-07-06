import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import joblib  # if you used joblib to save the model
from sklearn.preprocessing import StandardScaler

# Load model (edit this based on how you saved it)
model = xgb.XGBRegressor()
model.load_model("models/xgboost_baseline.json")

# Load test data
X = np.load("model_data/X.npy")
y = np.load("model_data/y.npy").reshape(-1, 1)

# Apply the same scaler (if you scaled X before training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WARNING: replace with actual fitted scaler if available

# Split (exact same as before)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Global feature importance (top 20)
shap.summary_plot(shap_values, X_test, max_display=20)
