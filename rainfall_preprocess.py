import pandas as pd
df = pd.read_csv("weather_dataset_2021_to_2025.csv")
df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
df = df.dropna(subset=["TIME"])
df["RAINFALL"] = df["RAINFALL"].fillna(0)
df = df[df["RAINFALL"] >= 0]
df["YEAR"] = df["TIME"].dt.year
df["MONTH"] = df["TIME"].dt.month
df["DAY"] = df["TIME"].dt.day
df = df.reset_index(drop=True)
print(df.head())
print(df.info())
