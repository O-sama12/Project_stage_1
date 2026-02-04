import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv("crop_yield_data_2021_2025_geo_region.csv")

crops = sorted(df["Crop"].unique())

cmap = plt.get_cmap("tab20", len(crops))

crop_colors = {
    crop: mcolors.to_hex(cmap(i))
    for i, crop in enumerate(crops)
}

india_map = folium.Map(
    location=[22.9734, 78.6569],
    zoom_start=5,
    tiles="cartodbpositron"
)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color=crop_colors[row["Crop"]],
        fill=True,
        fill_color=crop_colors[row["Crop"]],
        fill_opacity=0.7,
        popup=f"""
        <b>State:</b> {row['State']}<br>
        <b>Crop:</b> {row['Crop']}<br>
        <b>Yield:</b> {row['Yield']}
        """
    ).add_to(india_map)

india_map.save("india_crop_distribution.html")
