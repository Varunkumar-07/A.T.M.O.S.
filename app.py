import streamlit as st
from src.models import predict  # import your updated predict.py

st.title("🌦️ WeatherXAI Starter App")
st.write("Hello Varun! 🎉 Your Streamlit app is running successfully.")

st.subheader("Next Steps")
st.markdown("""
- Fetch real weather data  
- Train a model  
- Add explanations  
""")

# ===== Step 1: User inputs =====
st.subheader("Enter Weather Features for Prediction")
temperature_max = st.number_input("Temperature Max (°C)", value=30.0, step=0.1)
temperature_min = st.number_input("Temperature Min (°C)", value=22.0, step=0.1)
precipitation = st.number_input("Precipitation (mm)", value=0.0, step=0.1)

# ===== Step 2: Prediction button =====
if st.button("Predict Next-Day Max Temperature"):
    # Pass user inputs to the model
    temperature = predict.get_prediction(
        temperature_max=temperature_max,
        temperature_min=temperature_min,
        precipitation=precipitation
    )
    st.write(f"🌡️ Predicted next-day max temperature: {temperature}°C")
