# src/models/predict.py
import pandas as pd
import joblib

# 1. Load trained model
model = joblib.load("models/weather_model.pkl")
print("✅ Model loaded!")

def get_prediction(temperature_max=30.0, temperature_min=22.0, precipitation=0.0):
    """
    Returns predicted next-day max temperature.
    You can pass custom values for features if needed.
    """
    input_data = {
        "temperature_2m_max": [temperature_max],
        "temperature_2m_min": [temperature_min],
        "precipitation_sum": [precipitation]
    }

    df_input = pd.DataFrame(input_data)
    pred = model.predict(df_input)
    return round(pred[0], 2)  # return value instead of printing

# Optional: if you run this file directly, it still prints
if __name__ == "__main__":
    predicted_temp = get_prediction()
    print(f"🌡️ Predicted next-day max temperature: {predicted_temp}°C")
