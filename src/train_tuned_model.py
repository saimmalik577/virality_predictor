import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load data
X = np.load("model_data/X.npy")
y = np.load("model_data/y.npy")
print(f"✅ Loaded X: {X.shape}, y: {y.shape}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# Set up model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Grid search
print("🔍 Tuning hyperparameters with GridSearchCV...")
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("✅ Best parameters:", grid_search.best_params_)

# Evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n📊 Tuned Model Evaluation:")
print("MAE: ", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²:  ", round(r2, 4))

# Save model
os.makedirs("models", exist_ok=True)
best_model.save_model("models/xgboost_tuned.json")
print("✅ Tuned model saved to models/xgboost_tuned.json")
