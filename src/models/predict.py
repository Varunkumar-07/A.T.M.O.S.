# src/models/predict.py
import pandas as pd
import joblib
import os
import warnings

MODEL_PATH = os.getenv("WEATHER_MODEL_PATH", "models/weather_model.pkl")

def _load_artifact(path=MODEL_PATH):
    obj = joblib.load(path)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        features = obj.get("features")
    else:
        model = obj
        features = getattr(model, "feature_names_in_", None)

    if features is None:
        features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
        warnings.warn(f"Model artifact has no feature names. Falling back to: {features}")
    else:
        features = list(features)

    return model, features


# Load once at import
_model, _features = _load_artifact()
print(f"Model loaded. Features: {_features}")

def get_prediction(temperature_max=30.0, temperature_min=22.0, precipitation=0.0, windspeed=5.0, temp_max_lag1=None, temp_min_lag1=None):
    # Gather all potential inputs
    full_input = {
        "temperature_2m_max": float(temperature_max),
        "temperature_2m_min": float(temperature_min),
        "precipitation_sum": float(precipitation),
        "windspeed_10m_max": float(windspeed),
        "temp_max_lag1": float(temp_max_lag1 if temp_max_lag1 is not None else temperature_max),
        "temp_min_lag1": float(temp_min_lag1 if temp_min_lag1 is not None else temperature_min),
    }

    # Ensure all expected features exist; fill missing ones with 0.0
    df_input = pd.DataFrame([full_input])
    for f in _features:
        if f not in df_input.columns:
            df_input[f] = 0.0

    # Reorder & drop extras to exactly match training features
    df_input = df_input[_features]

    pred = _model.predict(df_input)
    return round(float(pred[0]), 2)



if __name__ == "__main__":
    p = get_prediction()
    print(f"Predicted next-day max temp: {p}°C")
