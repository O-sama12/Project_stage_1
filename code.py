import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt 

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("crop_yield_data_2021_2024.csv")

# ============================================
# FEATURE ENGINEERING
# ============================================

# Remove extreme outliers (top 1%)
df = df[df["Yield"] < df["Yield"].quantile(0.99)]

# Create new interaction features
df["Fert_per_Area"] = df["Fertilizer"] / (df["Area"] + 1)
df["Pest_per_Area"] = df["Pesticide"] / (df["Area"] + 1)
df["Prod_per_Area"] = df["Production"] / (df["Area"] + 1)

# Log-transform Yield to stabilize scale
df["Yield_log"] = np.log1p(df["Yield"])

# ============================================
# SELECT FEATURES / TARGET
# ============================================

target = "Yield_log"
X = df.drop(columns=["Yield", "Yield_log"])
y = df[target]

categorical_cols = ["Crop", "Season", "State"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# ============================================
# PREPROCESSING PIPELINE
# ============================================

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ============================================
# OPTIMIZED HGBR MODEL
# ============================================

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("hgb", HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=600,
            l2_regularization=0.1,
            min_samples_leaf=20,
            random_state=42
        ))
    ]
)

# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

# ============================================
# PREDICT & EVALUATE
# ============================================

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)         # reverse log transform
y_true = np.expm1(y_test)

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("\n📊 OPTIMIZED MODEL PERFORMANCE (HGBR)")
print("--------------------------------------")
print(f"R² Score:        {r2:.4f}")
print(f"MAE:             {mae:.4f}")
print(f"MSE:             {mse:.4f}")
print(f"RMSE:            {rmse:.4f}")

# ============================================
# ACTUAL VS PREDICTED PLOT
# ============================================

plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.6)

# Perfect prediction line
min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield (HGBR)")
plt.grid(True)
plt.show()

# ============================================
# FEATURE IMPORTANCE PLOT
# ============================================

result = permutation_importance(model, X_test, y_test, n_repeats=10)

plt.figure(figsize=(10, 5))
plt.barh(X.columns, result.importances_mean)
plt.title("Feature Importance")
plt.show()
