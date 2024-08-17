import streamlit as st
from streamlit_geolocation import streamlit_geolocation
import requests
import pandas as pd
from datetime import datetime, timedelta


# Function to get weather data from Visual Crossing API
def get_weather_data(lat, lon, start_date, end_date, api_key):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"

    params = {
        'unitGroup': 'metric',  # Use 'us' for Fahrenheit and 'metric' for Celsius
        'key': api_key,
        'include': 'days',  # Include daily data
        'elements': 'tempmax,tempmin,precip,windspeed'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['days']
    else:
        st.error(f"Error: {response.status_code}")
        return None


# Function to create a summary string from the weather data
def create_summary(weather_data, period_name):
    if not weather_data:
        return f"No data available for the {period_name}."

    temps_max = [day['tempmax'] for day in weather_data if 'tempmax' in day]
    temps_min = [day['tempmin'] for day in weather_data if 'tempmin' in day]
    precip = [day.get('precip', 0) for day in weather_data]
    wind_speeds = [day['windspeed'] for day in weather_data if 'windspeed' in day]

    summary = (
        f"Weather Summary for {period_name} is that, "
        f"Max Temp: {max(temps_max, default='N/A')}°C, Min Temp: {min(temps_min, default='N/A')}°C, "
        f"Total Precipitation: {sum(precip)} mm\n and "
        f"Average Wind Speed: {sum(wind_speeds) / len(wind_speeds) if wind_speeds else 'N/A'} km/h\n"
    )
    return summary


# API Key
api_key = 'WYJXSZL8MMXELRQL9U3R2P6WR'  # Replace with your Visual Crossing API key

# Streamlit app
st.title("Weather Summary")

location = streamlit_geolocation()

if location and location['latitude'] and location['longitude']:
    latitude = location['latitude']
    longitude = location['longitude']

    # Dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date_last_month = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Last 1 month
    start_date_next_15_days = datetime.now().strftime('%Y-%m-%d')
    end_date_next_15_days = (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')

    # Get historical weather data for the last 1 month
    historical_weather = get_weather_data(latitude, longitude, start_date_last_month, end_date, api_key)

    # Get weather forecast for the next 15 days
    forecast_weather = get_weather_data(latitude, longitude, start_date_next_15_days, end_date_next_15_days, api_key)

    # Create summaries
    historical_summary = create_summary(historical_weather, "Last 1 Month")
    forecast_summary = create_summary(forecast_weather, "Next 15 Days")

    # Display summaries in Streamlit text boxes
    st.subheader("Historical Weather Summary")
    st.text(historical_summary)

    st.subheader("Forecast Weather Summary")
    st.text(forecast_summary)
else:
    st.error("Location data is not available.")
