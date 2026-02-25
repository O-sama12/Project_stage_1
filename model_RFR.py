import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error
)
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("crop_yield_data_2021_2025_geo_region.csv")

# Separate features and target
X = df.drop(columns=["Yield"])
y = df["Yield"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Verify geo features
geo_features = [
    c for c in X.columns
    if "Latitude" in c or "Longitude" in c or "Region_" in c
]
print("Geo features used:", geo_features)

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Log-transform target (for skew handling)
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

# -------------------------------
# Random Forest Regressor Model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train_log)

# Predict
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Yield cannot be negative
y_pred = np.maximum(y_pred, 0)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

# sMAPE
eps = 1e-6
smape = np.mean(
    2 * np.abs(y_pred - y_test) /
    (np.abs(y_test) + np.abs(y_pred) + eps)
) * 100

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Results
print("📊 Random Forest Model Performance")
print("R² Score :", r2)
print("Adjusted R² :", adj_r2)
print("RMSE :", rmse)
print("MAE :", mae)
print("RMSLE :", rmsle)
print("sMAPE (%) :", smape)

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield (Random Forest)")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Yield")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted Yield (Random Forest)")
plt.tight_layout()
plt.show()