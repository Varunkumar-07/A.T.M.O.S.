import requests
import pandas as pd
import os

LAT, LON = 12.9716, 77.5946

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date=2015-01-01&end_date=2022-12-31"
    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max"
    f"&timezone=auto"
)

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data["daily"])

os.makedirs("data", exist_ok=True)

csv_path = "data/weather_history.csv"
df.to_csv(csv_path, index=False)
print(f"Weather data saved to {csv_path}")
