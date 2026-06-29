import requests
import pandas as pd

def get_weather_7days(latitude=12.97, longitude=77.59):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,"
        f"windspeed_10m_max,winddirection_10m_dominant,sunrise,sunset"
        f"&past_days=3"
        f"&forecast_days=4"   # today + next 3
        f"&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data['daily'])
    df["time"] = pd.to_datetime(df["time"])
    return df
