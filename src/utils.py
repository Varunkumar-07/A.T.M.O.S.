import requests


def get_coordinates(city: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    r = requests.get(url)
    data = r.json()
    if "results" in data and data["results"]:
        result = data["results"][0]
        return result["latitude"], result["longitude"], result["name"]
    return None, None, None
