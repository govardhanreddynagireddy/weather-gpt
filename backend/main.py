from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent.orchestrator import orchestrate
from backend.tools.weather_tool import get_weather


# =====================================================
# APP
# =====================================================

app=FastAPI(

    title="WeatherGPT+",

    description=(
        "Conversational AI for weather forecasting, "
        "risk analysis and alerts"
    ),

    version="1.0.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)


# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):

    message:str


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {

        "message":"WeatherGPT+ API is running",

        "status":"success"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():

    return {

        "status":"healthy"
    }


# =====================================================
# CURRENT WEATHER
# =====================================================

@app.get("/weather/{city}")
def weather(city:str):

    return get_weather(city)


# =====================================================
# CHAT
# =====================================================

@app.post("/chat")
def chat(request:ChatRequest):

    result=orchestrate(

        request.message
    )

    return result