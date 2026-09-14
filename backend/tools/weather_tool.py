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

    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url="https://api.openweathermap.org/data/2.5/weather"

    params={
        "q":city,
        "appid":API_KEY,
        "units":"metric"
    }

    response=requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data=response.json()

    return {
        "city":data["name"],
        "country":data["sys"]["country"],
        "temperature_celsius":data["main"]["temp"],
        "humidity_percent":data["main"]["humidity"],
        "wind_speed_kmh":round(data["wind"]["speed"]*3.6,2),
        "precipitation_mm":data.get("rain",{}).get("1h",0),
        "weather_description":data["weather"][0]["description"],
        "time":datetime.now().strftime("%H:%M")
    }


# =====================================================
# WEATHER FORECAST
# =====================================================

def get_weather_forecast(city,days_ahead=1):

    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url="https://api.openweathermap.org/data/2.5/forecast"

    params={
        "q":city,
        "appid":API_KEY,
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

    for item in data["list"]:

        forecast_time=datetime.fromtimestamp(
            item["dt"]
        )

        if forecast_time.date()==target_date:

            forecasts.append({
                "time":forecast_time.strftime("%H:%M"),
                "temperature":item["main"]["temp"],
                "feels_like":item["main"]["feels_like"],
                "humidity":item["main"]["humidity"],
                "wind_speed":item["wind"]["speed"]*3.6,
                "weather":item["weather"][0]["description"],
                "rain_probability":item.get("pop",0)*100,
                "rainfall":item.get(
                    "rain",
                    {}
                ).get(
                    "3h",
                    0
                )
            })

    if not forecasts:

        raise ValueError(
            "No forecast data available for the requested date."
        )

    return {
        "location":data["city"]["name"],
        "date":str(target_date),
        "forecasts":forecasts
    }