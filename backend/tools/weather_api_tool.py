import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("OPENWEATHER_API_KEY")


def get_weather(city):

    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url="https://api.openweathermap.org/data/2.5/weather"

    params={
        "q":city,
        "appid":API_KEY,
        "units":"metric"
    }

    response=requests.get(url,params=params,timeout=10)

    response.raise_for_status()

    data=response.json()

    return {
        "location":data["name"],
        "temperature":data["main"]["temp"],
        "humidity":data["main"]["humidity"],
        "wind_speed":data["wind"]["speed"],
        "weather":data["weather"][0]["description"],
        "rainfall":data.get("rain",{}).get("1h",0)
    }