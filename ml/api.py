import torch
import xarray as xr
import joblib
import requests
from rag.rag_tool import get_context
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from ml.model import WeatherGRU


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH="ml/data/processed/best_weather_gru.pth"
DATA_PATH="ml/data/processed/era5_merged.nc"

FEATURE_SCALER_PATH="ml/data/processed/feature_scaler.pkl"
TARGET_SCALER_PATH="ml/data/processed/target_scaler.pkl"

FEATURES=[
    "d2m",
    "t2m",
    "sp",
    "tp",
    "ssrd",
    "skt",
    "u10",
    "v10"
]

OLLAMA_URL="http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL="llama3.2:3b"

DEVICE=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI
# ============================================================

app=FastAPI(
    title="WeatherGPT GRU API",
    description="Weather prediction using GRU, Open-Meteo and Ollama",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message:str


# ============================================================
# LOAD GRU MODEL
# ============================================================

model=WeatherGRU(
    input_size=8,
    hidden_size=128,
    num_layers=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print()
print("================================")
print("WeatherGPT")
print("================================")

print("Model loaded successfully!")
print("Device:",DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD SCALERS
# ============================================================

feature_scaler=joblib.load(
    FEATURE_SCALER_PATH
)

target_scaler=joblib.load(
    TARGET_SCALER_PATH
)

print("Scalers loaded successfully!")


# ============================================================
# LOAD ERA5 DATA
# ============================================================

ds=xr.open_dataset(DATA_PATH)

df=ds[FEATURES].to_dataframe()

df=df.sort_index()

print("ERA5 records:",len(df))
print("Latest time:",df.index[-1])


# ============================================================
# GRU NEXT-HOUR PREDICTION
# ============================================================

def get_temperature_prediction():

    latest_df=df.iloc[-24:][FEATURES]

    if len(latest_df)!=24:

        raise ValueError(
            "Not enough data for prediction"
        )


    # Scale features

    latest_scaled=feature_scaler.transform(
        latest_df
    )


    # Convert to tensor

    x=torch.tensor(
        latest_scaled,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)


    # GRU prediction

    with torch.no_grad():

        prediction=model(x)


    # Convert to numpy

    prediction=(
        prediction
        .cpu()
        .numpy()
        .reshape(-1,1)
    )


    # Inverse scaling

    temperature_kelvin=(
        target_scaler
        .inverse_transform(prediction)[0][0]
    )


    # Kelvin → Celsius

    temperature_celsius=(
        temperature_kelvin-273.15
    )


    return round(
        float(temperature_celsius),
        2
    )


# ============================================================
# CITY WEATHER
# ============================================================

def get_city_weather(city):

    # --------------------------------------------------------
    # GEOCODING
    # --------------------------------------------------------

    geo_url=(
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geo_response=requests.get(
        geo_url,
        params={
            "name":city,
            "count":1,
            "language":"en",
            "format":"json"
        },
        timeout=10
    )

    geo_response.raise_for_status()

    geo_data=geo_response.json()

    if "results" not in geo_data:

        return {
            "error":f"City '{city}' not found"
        }


    location=geo_data["results"][0]

    latitude=location["latitude"]
    longitude=location["longitude"]

    city_name=location["name"]

    country=location.get(
        "country",
        ""
    )


    # --------------------------------------------------------
    # CURRENT WEATHER
    # --------------------------------------------------------

    weather_url=(
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_response=requests.get(
        weather_url,
        params={
            "latitude":latitude,
            "longitude":longitude,
            "current":(
                "temperature_2m,"
                "relative_humidity_2m,"
                "wind_speed_10m,"
                "precipitation"
            ),
            "timezone":"auto"
        },
        timeout=10
    )

    weather_response.raise_for_status()

    weather_data=weather_response.json()

    current=weather_data["current"]


    return {

        "city":city_name,

        "country":country,

        "latitude":latitude,

        "longitude":longitude,

        "time":current["time"],

        "temperature_celsius":
            current["temperature_2m"],

        "humidity_percent":
            current["relative_humidity_2m"],

        "wind_speed_kmh":
            current["wind_speed_10m"],

        "precipitation_mm":
            current["precipitation"]

    }


# ============================================================
# TOMORROW FORECAST
# ============================================================

def get_tomorrow_forecast(city):

    # --------------------------------------------------------
    # GEOCODING
    # --------------------------------------------------------

    geo_url=(
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geo_response=requests.get(
        geo_url,
        params={
            "name":city,
            "count":1,
            "language":"en",
            "format":"json"
        },
        timeout=10
    )

    geo_response.raise_for_status()

    geo_data=geo_response.json()

    if "results" not in geo_data:

        return {
            "error":f"City '{city}' not found"
        }


    location=geo_data["results"][0]

    latitude=location["latitude"]
    longitude=location["longitude"]

    city_name=location["name"]

    country=location.get(
        "country",
        ""
    )


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast_url=(
        "https://api.open-meteo.com/v1/forecast"
    )

    forecast_response=requests.get(
        forecast_url,
        params={
            "latitude":latitude,
            "longitude":longitude,
            "daily":(
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_sum,"
                "weather_code"
            ),
            "forecast_days":2,
            "timezone":"auto"
        },
        timeout=10
    )

    forecast_response.raise_for_status()

    forecast_data=forecast_response.json()

    daily=forecast_data["daily"]


    # Tomorrow = index 1

    weather_code=int(
        daily["weather_code"][1]
    )


    # Weather code description

    weather_description=(
        get_weather_description(weather_code)
    )


    return {

        "city":city_name,

        "country":country,

        "date":daily["time"][1],

        "max_temperature_celsius":
            daily["temperature_2m_max"][1],

        "min_temperature_celsius":
            daily["temperature_2m_min"][1],

        "precipitation_mm":
            daily["precipitation_sum"][1],

        "weather_code":
            weather_code,

        "weather_description":
            weather_description

    }


# ============================================================
# WEATHER CODE DESCRIPTION
# ============================================================

def get_weather_description(code):

    weather_codes={

        0:"Clear sky",

        1:"Mainly clear",

        2:"Partly cloudy",

        3:"Overcast",

        45:"Fog",

        48:"Depositing rime fog",

        51:"Light drizzle",

        53:"Moderate drizzle",

        55:"Dense drizzle",

        61:"Slight rain",

        63:"Moderate rain",

        65:"Heavy rain",

        71:"Slight snow",

        73:"Moderate snow",

        75:"Heavy snow",

        80:"Slight rain showers",

        81:"Moderate rain showers",

        82:"Violent rain showers",

        95:"Thunderstorm",

        96:"Thunderstorm with slight hail",

        99:"Thunderstorm with heavy hail"

    }

    return weather_codes.get(
        code,
        "Unknown weather"
    )


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt):

    response=requests.post(
        OLLAMA_URL,
        json={
            "model":OLLAMA_MODEL,
            "prompt":prompt,
            "stream":False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "WeatherGPT GRU API",

        "status":
            "running",

        "device":
            str(DEVICE),

        "llm":
            OLLAMA_MODEL

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model":
            "WeatherGRU",

        "llm":
            "Ollama",

        "llm_model":
            OLLAMA_MODEL,

        "device":
            str(DEVICE),

        "gpu":(
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )

    }


# ============================================================
# NEXT-HOUR GRU PREDICTION
# ============================================================

@app.get("/predict")
def predict():

    temperature=(
        get_temperature_prediction()
    )

    return {

        "latest_data_time":
            str(df.index[-1]),

        "predicted_temperature_celsius":
            temperature,

        "device":
            str(DEVICE)

    }


# ============================================================
# CURRENT CITY WEATHER
# ============================================================

@app.get("/weather/{city}")
def weather(city:str):

    return get_city_weather(city)


# ============================================================
# TOMORROW FORECAST
# ============================================================

@app.get("/forecast/{city}")
def forecast(city:str):

    return get_tomorrow_forecast(city)


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(request:ChatRequest):

    message=request.message.strip()

    message_lower=message.lower()


    # ========================================================
    # GET GRU TEMPERATURE
    # ========================================================

    temperature=get_temperature_prediction()


    # ========================================================
    # DETECT CITY
    # ========================================================

    city=None


    if " in " in message_lower:

        city=message_lower.split(
            " in ",
            1
        )[1]

        city=city.strip()

        city=city.replace("?","")

        city=city.replace(".","")

        city=city.strip()


    # ========================================================
    # TOMORROW FORECAST
    # ========================================================

    tomorrow_words=[
        "tomorrow",
        "next day",
        "next-day"
    ]


    is_tomorrow=any(
        word in message_lower
        for word in tomorrow_words
    )


    if is_tomorrow and city:

        try:

            forecast_data=(
                get_tomorrow_forecast(city)
            )


            if "error" not in forecast_data:

                # ------------------------------------------------
                # IMPORTANT:
                # We create the factual answer ourselves.
                # Ollama does NOT change weather values.
                # ------------------------------------------------

                answer=(
                    f"Tomorrow in "
                    f"{forecast_data['city']}, "
                    f"temperatures are expected to range "
                    f"from "
                    f"{forecast_data['min_temperature_celsius']}°C "
                    f"to "
                    f"{forecast_data['max_temperature_celsius']}°C. "
                    f"The weather is expected to be "
                    f"{forecast_data['weather_description'].lower()} "
                    f"with "
                    f"{forecast_data['precipitation_mm']} mm "
                    f"of precipitation."
                )


                return {

                    "question":
                        message,

                    "type":
                        "tomorrow_forecast",

                    "forecast":
                        forecast_data,

                    "answer":
                        answer

                }


        except Exception as e:

            print(
                "Tomorrow forecast error:",
                e
            )


    # ========================================================
    # CURRENT CITY WEATHER
    # ========================================================

    if city:

        try:

            city_weather=(
                get_city_weather(city)
            )


            if "error" not in city_weather:

                # ------------------------------------------------
                # Build answer ourselves to prevent hallucination.
                # ------------------------------------------------

                answer=(
                    f"Currently in "
                    f"{city_weather['city']}, "
                    f"the temperature is "
                    f"{city_weather['temperature_celsius']}°C "
                    f"with humidity at "
                    f"{city_weather['humidity_percent']}%. "
                    f"Wind speed is "
                    f"{city_weather['wind_speed_kmh']} km/h "
                    f"and precipitation is "
                    f"{city_weather['precipitation_mm']} mm."
                )


                return {

                    "question":
                        message,

                    "type":
                        "city_weather",

                    "weather":
                        city_weather,

                    "answer":
                        answer

                }


        except Exception as e:

            print(
                "City weather error:",
                e
            )


    # ========================================================
    # NEXT-HOUR PREDICTION
    # ========================================================

    if (
        "next hour" in message_lower
        or "next-hour" in message_lower
        or "predict" in message_lower
    ):

        answer=(
            f"The WeatherGRU model predicts "
            f"the next-hour temperature will be "
            f"{temperature}°C."
        )


        return {

            "question":
                message,

            "type":
                "gru_prediction",

            "predicted_temperature_celsius":
                temperature,

            "answer":
                answer

        }


    # ========================================================
    # GENERAL CHAT
    # ========================================================

    prompt=f"""
You are WeatherGPT, a helpful weather assistant.

The WeatherGRU model has predicted the next-hour
temperature as {temperature} degrees Celsius.

User message:
{message}

Answer naturally and conversationally.

Do not invent weather measurements.

If the user asks for a specific city's current weather,
tell them to ask something like:
"What is the weather in Kurnool?"

If they ask for tomorrow's weather, tell them to ask:
"What will the weather be tomorrow in Kurnool?"

Keep the answer short.
"""


    try:

        answer=ask_ollama(prompt)

    except Exception as e:

        print(
            "Ollama error:",
            e
        )

        answer=(
            "I'm having trouble connecting to "
            "the language model right now."
        )


    return {

        "question":
            message,

        "type":
            "general",

        "predicted_temperature_celsius":
            temperature,

        "answer":
            answer

    }