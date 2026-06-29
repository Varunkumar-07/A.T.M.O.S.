import requests
import pandas as pd

def get_weather_7days(latitude=12.97, longitude=77.59):
    """
    Fetch past 3 days, today, and next 3 days weather data (7-day window).
    """
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

    # Convert to DataFrame
    df = pd.DataFrame(data['daily'])
    df["time"] = pd.to_datetime(df["time"])
    return df
