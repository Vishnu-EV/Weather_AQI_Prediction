import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/aqi_model.pkl")

# Load dataset
data = pd.read_csv("dataset/city_day.csv")
data = data.dropna()

st.title("🌍 Air Quality Prediction System")

# City dropdown
cities = sorted(data["City"].unique())
city = st.selectbox("Select City", cities)

if st.button("Predict AQI"):
    city_data = data[data["City"] == city]
    avg_values = city_data[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]].mean()

    prediction = model.predict([avg_values])

    st.success(f"Predicted AQI for {city}: {prediction[0]:.2f}")
