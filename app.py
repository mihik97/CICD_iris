import streamlit as st
import joblib
import requests

# Load trained model
model = joblib.load("model/random_forest_model.pkl")

# OpenWeather API Configuration
OPENWEATHER_API_KEY = "709b78ab4fd51633c71ae1004f89279b"

def get_weather(city, state, country):
    """Fetch weather data from OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{state},{country}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        
        # Choose weather emoji based on temperature
        if temp > 30:
            temp_emoji = "🔥"
        elif temp > 20:
            temp_emoji = "☀️"
        elif temp > 10:
            temp_emoji = "🌤️"
        else:
            temp_emoji = "❄️"

        return f"{temp}°C {temp_emoji}", f"{humidity}% 💧"
    else:
        return "N/A", "N/A"

# Streamlit UI
st.set_page_config(page_title="ML Model Deployment", layout="wide")

# Left Column (Model Input)
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🌸 Iris Flower Classification")
    st.write("Enter feature values to predict the Iris species.")

    # User Input
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

    # Make Prediction
    if st.button("Predict"):
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(features)[0]
        st.success(f"Predicted Species: **{prediction}** 🌼")

# Right Column (Weather Widget)
with col2:
    st.title("☁️ Live Weather")
    
    country = st.text_input("Country", "India")
    state = st.text_input("State", "Haryana")
    city = st.text_input("City", "Gurgaon")

    if st.button("Get Weather"):
        temp, humidity = get_weather(city, state, country)
        st.metric(label="Temperature", value=temp)
        st.metric(label="Humidity", value=humidity)