from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model, imputer, and scaler
model = joblib.load('app/model/model.pkl')
imputer = joblib.load('app/model/imputer.pkl')
scaler = joblib.load('app/model/scaler.pkl')

# Define input data model
class WateringInput(BaseModel):
    soil_moisture: int
    temperature: int
    soil_humidity: int  # Maps to ' Soil Humidity' in dataset
    time: int
    air_temperature: float
    wind_speed: float
    air_humidity: float
    wind_gust: float
    pressure: float

@app.get("/health")
async def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(input_data: WateringInput):
    try:
        # Convert input to array, mapping to dataset column names
        input_array = np.array([[
            input_data.soil_moisture,
            input_data.temperature,
            input_data.soil_humidity,  # Maps to ' Soil Humidity'
            input_data.time,
            input_data.air_temperature,
            input_data.wind_speed,
            input_data.air_humidity,
            input_data.wind_gust,
            input_data.pressure
        ]])

        # Impute and scale input
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        result = "ON" if prediction == 1 else "OFF"

        return {"watering_status": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))