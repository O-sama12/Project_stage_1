import pandas as pd
df = pd.read_csv("crop_yield_data_2021_2025.csv")
# drop exact duplicate rows
df = df.drop_duplicates()
# remove clearly invalid values
df = df[
    (df["Area"] > 0) &
    (df["Annual_Rainfall"] > 0) &
    (df["Fertilizer"] >= 0) &
    (df["Pesticide"] >= 0) &
    (df["Yield"] > 0)
]
# (assuming Yield is the target)
df = df.drop(columns=["Production"])

def collapse_rare(series, min_freq=50):
    counts = series.value_counts()
    rare = counts[counts < min_freq].index
    return series.replace(rare, "OTHER")

df["Crop"] = collapse_rare(df["Crop"])
df["State"] = collapse_rare(df["State"])
df = df.sort_values("Crop_Year").reset_index(drop=True)

X = df.drop(columns=["Yield"])
y = df["Yield"]

print("Preprocessing complete")
print("Shape:", df.shape)
