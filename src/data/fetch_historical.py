import requests
import pandas as pd

# Example: Bangalore coordinates (you can change later)
lat, lon = 12.9716, 77.5946

# Request data from Open-Meteo API
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={lon}"
    f"&start_date=2015-01-01&end_date=2022-12-31"
    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
    f"&timezone=auto"
)

response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data["daily"])

# Save to CSV
csv_path = "data/weather_history.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Weather data saved to {csv_path}")

# Preview
print("\nSample data:")
print(df.head())

# Ready for ML pipeline
X = df[["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]]
y = df["temperature_2m_max"].shift(-1)  # Example target: next-day max temp

print("\n✅ Features (X) and Target (y) prepared for training!")
print("X shape:", X.shape)
print("y shape:", y.shape)
