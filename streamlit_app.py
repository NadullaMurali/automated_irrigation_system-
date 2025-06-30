import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint
API_URL = "http://localhost:8000/predict"

st.title("Crop Watering Prediction System")

st.write("Enter environmental parameters to predict whether watering is needed.")

# Input fields
soil_moisture = st.slider("Soil Moisture (%)", 1, 90, 45)
temperature = st.slider("Temperature (°C)", 0, 45, 22)
soil_humidity = st.slider("Soil Humidity (%)", 20, 70, 45)  # Maps to ' Soil Humidity' in dataset
time = st.slider("Time (arbitrary units)", 0, 110, 55)
air_temperature = st.slider("Air Temperature (°C)", 11.0, 45.0, 24.0)
wind_speed = st.slider("Wind Speed (Km/h)", 0.0, 31.0, 10.0)
air_humidity = st.slider("Air Humidity (%)", 0.0, 96.0, 58.0)
wind_gust = st.slider("Wind Gust (Km/h)", 0.0, 133.0, 41.0)
pressure = st.slider("Pressure (KPa)", 100.0, 101.8, 101.1)

if st.button("Predict Watering Status"):
    # Prepare input data
    input_data = {
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "soil_humidity": soil_humidity,  # Maps to ' Soil Humidity' in dataset
        "time": time,
        "air_temperature": air_temperature,
        "wind_speed": wind_speed,
        "air_humidity": air_humidity,
        "wind_gust": wind_gust,
        "pressure": pressure
    }

    # Make API request
    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        result = response.json()
        st.success(f"Watering Status: {result['watering_status']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {str(e)}")