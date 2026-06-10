import joblib

# Load saved model
model = joblib.load("models/aqi_model.pkl")

# Sample input: [PM2.5, PM10, NO2, SO2, CO, O3]
sample = [[80, 120, 30, 10, 1.2, 25]]

prediction = model.predict(sample)

print("Predicted AQI:", prediction[0])
