from fastapi import FastAPI
from pydantic import BaseModel

from backend.agent.orchestrator import orchestrate


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
# CHAT
# =====================================================

@app.post("/chat")
def chat(request:ChatRequest):

    result=orchestrate(
        request.message
    )

    return result