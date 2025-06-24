import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# Define dataset file paths
WEATHER_DATASET = "weather_data.csv"
BREAST_CANCER_DATASET = "breast-cancer.csv"
HOUSE_PRICE_DATASET = "BostonHousing.csv"
CRICKET_SCORE_DATASET = "cricket_scores.csv"

# Load or Train Models
def load_or_train_models():
    if not os.path.exists("weather_data.pkl"):
        df = pd.read_csv(WEATHER_DATASET)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "weather_datal.pkl")

    if not os.path.exists("breast-caancer.pkl"):
        df = pd.read_csv(BREAST_CANCER_DATASET)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "breast-cancer.pkl")

    if not os.path.exists("BostonHousing.pkl"):
        df = pd.read_csv(HOUSE_PRICE_DATASET)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "BostonHousing.pkl")

    if not os.path.exists("cricket_scores.pkl"):
        df = pd.read_csv(CRICKET_SCORE_DATASET)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, "cricket_scores.pkl")

# Prediction Functionsnnṇ
def weather_prediction(features):
    model = joblib.load("weather_data.pkl")
    return f"Predicted Temperature: {model.predict([features])[0]:.2f}°C"

def breast_cancer_prediction(features):
    model = joblib.load("breast-cancer.pkl")
    return "Malignant" if model.predict([features])[0] == 0 else "Benign"

def house_price_prediction(features):
    model = joblib.load("BostonHousing.pkl")
    return f"Predicted House Price: ${model.predict([features])[0] * 100000:.2f}"

def cricket_score_prediction(features):
    model = joblib.load("cricket_scores.pkl")
    return f"Predicted Cricket Score: {int(model.predict([features])[0])}"
def spam_prediction(features):
    model = joblib.load("spam.pkl")
    return "spam" if model.predict([features])[0] == 'spam' else "ham"

# Streamlit UI
st.title("🔮 Multi-Purpose Prediction System")

# Sidebar for Model Selection
option = st.sidebar.radio("Select a Prediction Model", 
                          ["🌦️ Weather Prediction", "🩺 Breast Cancer Prediction", 
                           "🏡 House Price Prediction", "🏏 Cricket Score Prediction"])

# Weather Prediction
if option == "🌦️ Weather Prediction":
    st.subheader("☁️ Enter Weather Data:")
    st.write("**Required Features:**")
    st.write("- Humidity (%)\n- Wind Speed (m/s)\n- Pressure (hPa)\n- Yesterday's Temperature (°C)\n- Cloud Cover (%)")

    humidity = st.number_input("🌡️ Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input("💨 Wind Speed (m/s)", min_value=0.0, value=5.0)
    pressure = st.number_input("🔵 Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
    temperature_prev = st.number_input("🌡️ Yesterday's Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
    cloud_cover = st.number_input("☁️ Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)

    if st.button("🌡️ Predict Temperature"):
        features = [humidity, wind_speed, pressure, temperature_prev, cloud_cover]
        st.success(weather_prediction(features))

# Breast Cancer Prediction
elif option == "🩺 Breast Cancer Prediction":
    st.subheader("🔬 Enter Features for Breast Cancer Prediction:")
    st.write("**Required Features:** 30 Features from Mammography Scan")
    MeanRadius = st.number_input("Mean Radius")
    MeanPerimeter = st.number_input(" Mean Perimeter")
    MeanArea = st.number_input("Mean Area")
    MeanConcavity = st.number_input("Mean Concavity")
    MeanConcavePoints = st.number_input("Mean Concave Points")
    WorstRadius = st.number_input("Worst Radius")
    WorstPerimeter = st.number_input("Worst Perimeter")
    WorstArea = st.number_input("Worst Area")
    WorstConcavity = st.number_input("Worst Concavity")
    WorstConcavePoints = st.number_input(" Worst Concave Points")

    features = [MeanRadius, MeanPerimeter, MeanArea, MeanConcavity, MeanConcavePoints, WorstRadius, WorstPerimeter, WorstArea, WorstConcavity, WorstConcavePoints]

    if st.button("🩺 Predict Cancer Type"):
        st.success(f"Breast Cancer Prediction: {breast_cancer_prediction(features)}")

# House Price Prediction
elif option == "🏡 House Price Prediction":
    st.subheader("🏠 Enter Features for House Price Prediction:")
    st.write("**Required Features:**")
    st.write("- Median Income ($1000s)\n- House Age\n- Number of Rooms\n- Number of Bedrooms\n- Population in Block\n- Occupants per Household\n- Latitude\n- Longitude")

    income = st.number_input("💰 Median Income ($1000s)", min_value=0.0, value=5.0)
    house_age = st.number_input("🏠 House Age (Years)", min_value=0.0, value=20.0)
    num_rooms = st.number_input("🛏️ Number of Rooms", min_value=1, value=5)
    num_bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, value=3)
    population = st.number_input("🏙️ Population in Block", min_value=1, value=500)
    occupants = st.number_input("👥 Occupants per Household", min_value=1.0, value=2.5)
    latitude = st.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, value=34.0)
    longitude = st.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, value=-118.0)

    if st.button("🏡 Predict House Price"):
        features = [income, house_age, num_rooms, num_bedrooms, population, occupants, latitude, longitude]
        st.success(house_price_prediction(features))

# Cricket Score Prediction
elif option == "🏏 Cricket Score Prediction":
    st.subheader("🏏 Enter Cricket Match Data:")
    st.write("**Required Features:**")
    st.write("- Overs Bowled\n- Runs Scored\n- Wickets Fallen\n- Current Run Rate\n- Opponent Strength (1-10)")

    overs = st.number_input("🔄 Overs Bowled", min_value=0.0, max_value=50.0, value=20.0)
    runs = st.number_input("🏏 Runs Scored", min_value=0, value=100)
    wickets = st.number_input("❌ Wickets Fallen", min_value=0, max_value=10, value=3)
    run_rate = st.number_input("📊 Current Run Rate", min_value=0.0, max_value=10.0, value=4.5)
    opponent_strength = st.number_input("💪 Opponent Strength (1-10)", min_value=1.0, max_value=10.0, value=7.0)

    if st.button("🏏 Predict Cricket Score"):
        features = [overs, runs, wickets, run_rate, opponent_strength]
        st.success(cricket_score_prediction(features))
