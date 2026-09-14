import os
import requests
from dotenv import load_dotenv
from datetime import datetime,timedelta

load_dotenv()

API_KEY=os.getenv("OPENWEATHER_API_KEY")


# =====================================================
# CURRENT WEATHER
# =====================================================

def get_weather(city):

    if not city or str(city).strip().lower()=="undefined":
        raise ValueError("Invalid city")

    city = str(city).strip()
    api_key = os.getenv("OPENWEATHER_API_KEY") or API_KEY

    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url="https://api.openweathermap.org/data/2.5/weather"

    params={
        "q":city,
        "appid":api_key,
        "units":"metric"
    }

    response=requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data=response.json()

    rain_data = data.get("rain") or {}
    precipitation = 0.0
    if isinstance(rain_data, dict):
        precipitation = rain_data.get("1h", rain_data.get("3h", 0.0)) or 0.0

    weather_desc = "Clear"
    if data.get("weather") and len(data["weather"]) > 0:
        weather_desc = data["weather"][0].get("description", "Clear")

    return {
        "city":data.get("name") or city,
        "country":data.get("sys",{}).get("country","IN"),
        "temperature_celsius":round(float(data.get("main",{}).get("temp", 0.0)), 2),
        "humidity_percent":int(data.get("main",{}).get("humidity", 0)),
        "wind_speed_kmh":round(float(data.get("wind",{}).get("speed", 0.0))*3.6, 2),
        "weather_description":weather_desc,
        "precipitation_mm":round(float(precipitation), 2),
        "time":datetime.now().strftime("%H:%M")
    }


# =====================================================
# WEATHER FORECAST
# =====================================================

def get_weather_forecast(city,days_ahead=1):

    if not city or str(city).strip().lower()=="undefined":
        raise ValueError("Invalid city")

    city = str(city).strip()
    api_key = os.getenv("OPENWEATHER_API_KEY") or API_KEY

    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url="https://api.openweathermap.org/data/2.5/forecast"

    params={
        "q":city,
        "appid":api_key,
        "units":"metric"
    }

    response=requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data=response.json()

    target_date=(
        datetime.now()+timedelta(days=days_ahead)
    ).date()

    forecasts=[]

    for item in data.get("list", []):

        forecast_time=datetime.fromtimestamp(
            item["dt"]
        )

        if forecast_time.date()==target_date:

            rain_data=item.get("rain") or {}
            rain_val=0.0
            if isinstance(rain_data, dict):
                rain_val=rain_data.get("3h", 0.0) or 0.0

            w_desc = "Clear"
            if item.get("weather") and len(item["weather"]) > 0:
                w_desc = item["weather"][0].get("description", "Clear")

            forecasts.append({
                "time":forecast_time.strftime("%H:%M"),
                "temperature":round(float(item.get("main",{}).get("temp", 0.0)), 2),
                "feels_like":round(float(item.get("main",{}).get("feels_like", 0.0)), 2),
                "humidity":int(item.get("main",{}).get("humidity", 0)),
                "wind_speed":round(float(item.get("wind",{}).get("speed", 0.0))*3.6, 2),
                "weather":w_desc,
                "rain_probability":round(float(item.get("pop", 0.0))*100, 1),
                "rainfall":round(float(rain_val), 2)
            })

    if not forecasts:
        raise ValueError(
            "No forecast data available for the requested date."
        )

    return {
        "city":data.get("city",{}).get("name") or city,
        "country":data.get("city",{}).get("country", "IN"),
        "date":str(target_date),
        "forecasts":forecasts
    }