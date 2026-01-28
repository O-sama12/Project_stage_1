import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("crop_yield_data_2021_2025.csv")

# Separate features and target
X = df.drop(columns=["Yield"])
y = df["Yield"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize model
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=300,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

smape = np.mean(
    2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))
) * 100

# Adjusted R²
n = X_test.shape[0] # number of samples
p = X_test.shape[1] # number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("📊 Model Performance")
print("R² Score :", r2)
print("Adjusted R² :", adj_r2)
print("RMSE :", rmse)
print("MAE :", mae)
print("sMAPE (%) :", smape)

#Actual vs Predicted plot

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha = 0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle = "--"
)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield using HistGradientBoostRegressor")
plt.tight_layout()
plt.show()

#Residual vs Predicted plot

residual = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha = 0.6)
plt.axhline(y = 0, linestyle = "--")
plt.xlabel("Predicted Yield")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted Yield")
plt.tight_layout()
plt.show()

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(
importance_df["Feature"][:10][::-1],
importance_df["Importance"][:10][::-1]
)


plt.xlabel("Permutation Importance")
plt.title("Top 10 Feature Importances")


plt.tight_layout()
plt.show()