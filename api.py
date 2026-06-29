from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json, joblib, pandas as pd, requests
from src.models.fetch_weather import get_weather_7days
from src.models import predict as predictor
from src.utils import get_coordinates

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



with open("models/metrics.json") as f:
    metrics = json.load(f)
with open("models/feature_importance.json") as f:
    feature_importance = json.load(f)
with open("models/model_comparison.json") as f:
    model_comparison = json.load(f)
with open("models/cv_results.json") as f:
    cv_results = json.load(f)
rain_artifact = joblib.load("models/rain_classifier.pkl")
rain_model = rain_artifact["model"]


from datetime import timedelta

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


@app.get("/weather")
def get_weather(city: str = Query(...)):
    lat, lon, resolved_name = get_coordinates(city)
    if not lat:
        return {"error": "City not found"}
    df = get_weather_7days(lat, lon)
    today = df.iloc[3]
    return {
        "city": resolved_name,
        "lat": lat,
        "lon": lon,
        "today": {
            "temp_max": round(float(today["temperature_2m_max"]), 1),
            "temp_min": round(float(today["temperature_2m_min"]), 1),
            "precipitation": round(float(today["precipitation_sum"]), 1),
            "windspeed": round(float(today["windspeed_10m_max"]), 1),
            "sunrise": str(today["sunrise"]),
            "sunset": str(today["sunset"]),
        },
        "trend": [
            {
                "date": str(row["time"])[:10],
                "temp_max": round(float(row["temperature_2m_max"]), 1),
                "temp_min": round(float(row["temperature_2m_min"]), 1),
            }
            for _, row in df.iterrows()
        ]
    }


@app.get("/predict")
def get_predictions(city: str = Query(...)):
    lat, lon, resolved_name = get_coordinates(city)
    if not lat:
        return {"error": "City not found"}
    df = get_weather_7days(lat, lon)
    today = df.iloc[3]

    temp_max = float(today["temperature_2m_max"])
    temp_min = float(today["temperature_2m_min"])
    precipitation = float(today["precipitation_sum"])
    windspeed = float(today["windspeed_10m_max"])
    prev_max = temp_max
    prev_min = temp_min
    today_date = today["time"]

    predictions = []
    for i in range(1, 8):
        pred_temp = predictor.get_prediction(
            temperature_max=temp_max,
            temperature_min=temp_min,
            precipitation=precipitation,
            windspeed=windspeed,
            temp_max_lag1=prev_max,
            temp_min_lag1=prev_min
        )
        if precipitation > 5.0:
            condition = "Rainy"
        elif pred_temp > 35:
            condition = "Hot"
        elif pred_temp > 28:
            condition = "Sunny"
        elif pred_temp > 20:
            condition = "Partly Cloudy"
        else:
            condition = "Cloudy"

        predictions.append({
            "date": (today_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "temp_max": pred_temp,
            "condition": condition,
            "confidence_range": metrics["MAE"]
        })
        prev_max = temp_max
        prev_min = temp_min
        temp_max = pred_temp

    rain_input = pd.DataFrame([{
        "temperature_2m_max": float(today["temperature_2m_max"]),
        "temperature_2m_min": float(today["temperature_2m_min"]),
        "precipitation_sum": float(today["precipitation_sum"]),
        "windspeed_10m_max": float(today["windspeed_10m_max"]),
        "temp_max_lag1": float(today["temperature_2m_max"]),
        "temp_min_lag1": float(today["temperature_2m_min"]),
    }])
    rain_pred = int(rain_model.predict(rain_input)[0])
    rain_prob = round(float(rain_model.predict_proba(rain_input)[0][1]) * 100, 1)

    return {
        "city": resolved_name,
        "predictions": predictions,
        "rain_tomorrow": rain_pred,
        "rain_probability": rain_prob,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_comparison": model_comparison,
        "cv_results": cv_results,
    }


@app.get("/autocomplete")
def autocomplete(query: str = Query(...)):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=6"
    r = requests.get(url)
    data = r.json()
    results = []
    if "results" in data:
        for item in data["results"]:
            results.append({
                "name": item.get("name", ""),
                "admin1": item.get("admin1", ""),
                "country": item.get("country", "")
            })
    return {"results": results}


@app.get("/compare")
def compare_cities(city_a: str = Query(...), city_b: str = Query(...)):
    def fetch(city):
        lat, lon, name = get_coordinates(city)
        if not lat:
            return None
        df = get_weather_7days(lat, lon)
        today = df.iloc[3]
        return {
            "city": name,
            "temp_max": round(float(today["temperature_2m_max"]), 1),
            "temp_min": round(float(today["temperature_2m_min"]), 1),
            "precipitation": round(float(today["precipitation_sum"]), 1),
            "windspeed": round(float(today["windspeed_10m_max"]), 1),
            "trend": [
                {
                    "date": str(row["time"])[:10],
                    "temp_max": round(float(row["temperature_2m_max"]), 1),
                }
                for _, row in df.iterrows()
            ]
        }
    return {"city_a": fetch(city_a), "city_b": fetch(city_b)}
