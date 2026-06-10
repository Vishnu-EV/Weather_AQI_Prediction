import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load dataset
data = pd.read_csv("dataset/city_day.csv")

# Drop missing values
data = data.dropna()

# Features (Inputs)
X = data[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]

# Target (Output AQI)
y = data["AQI"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/aqi_model.pkl")
print("Model saved successfully!")
