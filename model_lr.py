import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

# Load dataset
df = pd.read_csv("crop_yield_data_1997_to_2025_geo_region.csv")

# 🔍 Check columns
print(df.columns)

# -----------------------------
# 🎯 Target Variable
# -----------------------------
y = df["Yield"]   # assuming column name is 'Yield'

# -----------------------------
# 📊 Feature Selection
# -----------------------------
X = df.drop(columns=["Yield"])

# -----------------------------
# 🧹 Handle Categorical Columns
# -----------------------------
categorical_cols = ["Crop", "Season", "State", "Region"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# One-hot encoding for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# -----------------------------
# ⚙️ Create Pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------
# ✂️ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 🚀 Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 📈 Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 📊 Evaluation
# -----------------------------
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# RMSLE (VERY important for skewed targets)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

# Safe sMAPE (epsilon avoids explosion near zero)
eps = 1e-6
smape = np.mean(
    2 * np.abs(y_pred - y_test) /
    (np.abs(y_test) + np.abs(y_pred) + eps)
) * 100

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("📊 Model Performance")
print("R² Score :", r2)
print("Adjusted R² :", adj_r2)
print("RMSE :", rmse)
print("MAE :", mae)
print("RMSLE :", rmsle)
print("sMAPE (%) :", smape)