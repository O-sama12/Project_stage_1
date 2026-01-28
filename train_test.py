import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
crop_df = pd.read_csv("crop_yield_data_2021_2025.csv")
weather_df = pd.read_csv("weather_dataset_2021_to_2025.csv")

# Split features and target (last column as target)
X_crop = crop_df.iloc[:, :-1]
y_crop = crop_df.iloc[:, -1]

X_weather = weather_df.iloc[:, :-1]
y_weather = weather_df.iloc[:, -1]

# Train-test split (80-20)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_weather, y_weather, test_size=0.2, random_state=42
)

# Print results
print("Crop Dataset:")
print("Total rows: ", crop_df.shape[0])
print("Train shape: ", Xc_train.shape)
print("Test shape: ", Xc_test.shape)

print("\nWeather Dataset:")
print("Total rows: ", weather_df.shape[0])
print("Train shape:", Xw_train.shape)
print("Test shape: ", Xw_test.shape)
