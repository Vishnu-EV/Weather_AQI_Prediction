import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("dataset/city_day.csv")
data = data.dropna()

# Load model
model = joblib.load("models/aqi_model.pkl")

# Select City
city = input("Enter City Name: ")

city_data = data[data["City"] == city]

if city_data.empty:
    print("City not found!")
else:
    # Take average pollutant values
    avg_values = city_data[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]].mean()

    prediction = model.predict([avg_values])
    print(f"Predicted AQI for {city}: {prediction[0]:.2f}")
def aqi_category(aqi):
    if aqi <= 50:
        return "Good 😊"
    elif aqi <= 100:
        return "Moderate 🙂"
    elif aqi <= 200:
        return "Poor 😷"
    elif aqi <= 300:
        return "Very Poor 🚨"
    else:
        return "Severe ☠️"
