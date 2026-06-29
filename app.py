import json
import joblib
import streamlit as st
import pandas as pd
from datetime import timedelta
from src.models import predict
from src.models.fetch_weather import get_weather_7days
from src.utils import get_coordinates

try:
    rain_artifact = joblib.load("models/rain_classifier.pkl")
    rain_model = rain_artifact["model"]
    rain_features = rain_artifact["features"]
    with open("models/rain_metrics.json") as f:
        rain_metrics = json.load(f)
except FileNotFoundError:
    rain_model = None
    rain_metrics = {}

# -------------------------------
# 🌟 Page Config + Custom CSS
# -------------------------------
st.set_page_config(
    page_title="🌦️ 7-Day Weather Forecast",
    page_icon="⛅",
    layout="wide",
)

try:
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.write("**Model:** Random Forest (100 trees)")
    st.sidebar.write("**Features:** Temp Max/Min, Precipitation, Windspeed + Lag features")
    st.sidebar.write(f"**MAE:** {metrics['MAE']}°C")
    st.sidebar.write(f"**RMSE:** {metrics['RMSE']}°C")
    st.sidebar.markdown("---")
    st.sidebar.write("📡 Data: Open-Meteo API (free, no key)")
except FileNotFoundError:
    pass

if rain_model:
    st.sidebar.markdown("### 🌧️ Rain Classifier")
    st.sidebar.write(f"**Accuracy:** {rain_metrics.get('accuracy', 'N/A')}")
    st.sidebar.write(f"**Precision:** {rain_metrics.get('rain_precision', 'N/A')}")
    st.sidebar.write(f"**Recall:** {rain_metrics.get('rain_recall', 'N/A')}")

try:
    with open("models/model_comparison.json") as f:
        mc = json.load(f)
    st.sidebar.markdown("### 📊 Model Comparison")
    for model_name, scores in mc.items():
        st.sidebar.write(f"**{model_name}:** MAE {scores['MAE']} | RMSE {scores['RMSE']}")
except FileNotFoundError:
    pass

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f7ff, #ffffff);
        padding: 2rem;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #1565c0;
        font-weight: 700;
    }
    .stButton button {
        background-color: #42a5f5;
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #1e88e5;
        color: #e3f2fd;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🏙️ App Title
# -------------------------------
st.title("🌦️ 7-Day Real-Time Weather Forecast")

city = st.text_input("🏙️ Enter city name", "Bangalore")

# -------------------------------
# 🚀 Main Logic
# -------------------------------
if city:
    lat, lon, _ = get_coordinates(city)
    if lat and lon:
        st.info(f"📍 Showing weather for **{city}** ({lat}, {lon})")

        # ✅ Fetch weather data
        df_weather = get_weather_7days(lat, lon)

        if not df_weather.empty:
            # ---------------------------------
            # 📊 Today's Data (4th row = today)
            # ---------------------------------
            today_data = df_weather.iloc[3]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡️ Max Temp", f"{today_data['temperature_2m_max']:.1f}°C")
            col2.metric("❄️ Min Temp", f"{today_data['temperature_2m_min']:.1f}°C")
            col3.metric("🌧️ Precipitation", f"{today_data['precipitation_sum']:.1f} mm")
            col4.metric("💨 Max Wind", f"{today_data['windspeed_10m_max']:.1f} km/h")

            # ---------------------------------
            # 📈 7-Day Temperature Trends
            # ---------------------------------
            st.subheader("📈 Temperature Trends (Past 3 + Today + Next 3 Days)")
            st.line_chart(
                df_weather.set_index("time")[["temperature_2m_max", "temperature_2m_min"]]
            )

            # ---------------------------------
            # 🌅 Sunrise & Sunset
            # ---------------------------------
            st.subheader("🌅 Sunrise & Sunset Times")
            st.dataframe(df_weather[["time", "sunrise", "sunset"]])

            # ---------------------------------
            # 🔮 ML Predictions (next 3 days)
            # ---------------------------------
            if st.button("🔮 Predict Next 7 Days (ML Model)"):
                latest_date = today_data["time"]
                temp_max = today_data["temperature_2m_max"]
                temp_min = today_data["temperature_2m_min"]
                precipitation = today_data["precipitation_sum"]
                windspeed = float(today_data["windspeed_10m_max"])
                prev_max = float(today_data["temperature_2m_max"])
                prev_min = float(today_data["temperature_2m_min"])

                predictions = []
                for i in range(1, 8):
                    pred_temp = predict.get_prediction(
                        temperature_max=temp_max,
                        temperature_min=temp_min,
                        precipitation=precipitation,
                        windspeed=windspeed,
                        temp_max_lag1=prev_max,
                        temp_min_lag1=prev_min
                    )
                    next_day = latest_date + timedelta(days=i)
                    if precipitation > 5.0:
                        condition = "🌧️ Rainy"
                    elif pred_temp > 35:
                        condition = "🔥 Hot"
                    elif pred_temp > 28:
                        condition = "☀️ Sunny"
                    elif pred_temp > 20:
                        condition = "⛅ Partly Cloudy"
                    else:
                        condition = "🌥️ Cloudy"
                    predictions.append({
                        "Date": next_day.strftime("%Y-%m-%d"),
                        "Predicted Max Temp (°C)": pred_temp,
                        "Condition": condition,
                        "Confidence Range": f"± {metrics['MAE']}°C"
                    })
                    prev_max = temp_max
                    prev_min = temp_min
                    temp_max = pred_temp

                df_pred = pd.DataFrame(predictions)

                # ✅ Merge real + predicted
                df_final = pd.concat([
                    df_weather[["time", "temperature_2m_max"]],
                    df_pred.rename(columns={"Date": "time", "Predicted Max Temp (°C)": "temperature_2m_max"}).assign(time=pd.to_datetime(df_pred["Date"]))
                ], ignore_index=True)

                st.subheader("🔮 Final 7-Day Forecast with Prediction")
                st.line_chart(df_final.set_index("time")[["temperature_2m_max"]])
                try:
                    with open("models/feature_importance.json") as f:
                        fi = json.load(f)
                    st.subheader("📊 Feature Importance")
                    st.bar_chart(fi)
                except FileNotFoundError:
                    pass
                if rain_model:
                    rain_input = pd.DataFrame([{
                        "temperature_2m_max": float(today_data["temperature_2m_max"]),
                        "temperature_2m_min": float(today_data["temperature_2m_min"]),
                        "precipitation_sum": float(today_data["precipitation_sum"]),
                        "windspeed_10m_max": float(today_data["windspeed_10m_max"]),
                        "temp_max_lag1": float(today_data["temperature_2m_max"]),
                        "temp_min_lag1": float(today_data["temperature_2m_min"]),
                    }])
                    rain_pred = rain_model.predict(rain_input)[0]
                    rain_prob = rain_model.predict_proba(rain_input)[0][1]
                    rain_label = "🌧️ Rain expected tomorrow" if rain_pred == 1 else "☀️ No rain expected tomorrow"
                    st.info(f"{rain_label} (confidence: {round(rain_prob * 100, 1)}%)")

                st.subheader("🔮 ML Predictions — Next 7 Days")
                st.dataframe(df_pred)
                csv = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name="weather_predictions.csv",
                    mime="text/csv"
                )

    else:
        st.error("❌ Could not find that city. Try another one.")

st.markdown("---")
st.subheader("🌍 Compare Two Cities")

col_a, col_b = st.columns(2)
city_a = col_a.text_input("🏙️ City A", "London")
city_b = col_b.text_input("🏙️ City B", "Tokyo")

if st.button("🔍 Compare Cities"):
    lat_a, lon_a, _ = get_coordinates(city_a)
    lat_b, lon_b, _ = get_coordinates(city_b)

    if lat_a and lat_b:
        df_a = get_weather_7days(lat_a, lon_a)
        df_b = get_weather_7days(lat_b, lon_b)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 📍 {city_a}")
            today_a = df_a.iloc[3]
            st.metric("Max Temp", f"{today_a['temperature_2m_max']:.1f}°C")
            st.metric("Min Temp", f"{today_a['temperature_2m_min']:.1f}°C")
            st.metric("Precipitation", f"{today_a['precipitation_sum']:.1f} mm")
            st.metric("Max Wind", f"{today_a['windspeed_10m_max']:.1f} km/h")

        with col2:
            st.markdown(f"### 📍 {city_b}")
            today_b = df_b.iloc[3]
            st.metric("Max Temp", f"{today_b['temperature_2m_max']:.1f}°C")
            st.metric("Min Temp", f"{today_b['temperature_2m_min']:.1f}°C")
            st.metric("Precipitation", f"{today_b['precipitation_sum']:.1f} mm")
            st.metric("Max Wind", f"{today_b['windspeed_10m_max']:.1f} km/h")

        st.subheader("📈 Temperature Trend Comparison")
        df_a_plot = df_a.set_index("time")[["temperature_2m_max"]].rename(
            columns={"temperature_2m_max": city_a}
        )
        df_b_plot = df_b.set_index("time")[["temperature_2m_max"]].rename(
            columns={"temperature_2m_max": city_b}
        )
        df_combined = df_a_plot.join(df_b_plot, how="outer")
        st.line_chart(df_combined)
    else:
        st.error("❌ Could not resolve one or both cities. Try different names.")
