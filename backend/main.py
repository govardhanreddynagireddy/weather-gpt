# =====================================================
# ENVIRONMENT GUARD — MUST RUN BEFORE torch / faiss /
# sentence-transformers ARE IMPORTED ANYWHERE IN THE
# PROCESS (directly or via orchestrator -> gru_tool /
# rag_tool). This prevents the native OpenMP crash
# ("OMP: Error #15: Initializing libiomp5md.dll, but
# found libiomp5md.dll already initialized") that
# happens on Windows when PyTorch's MKL runtime and
# FAISS's OpenMP runtime both try to init in one
# process. Without this, the process aborts natively
# and no Python try/except can catch it.
# =====================================================

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Guard against native access violation in PyTorch CUDA stream capture on CPU/Windows
import torch
if hasattr(torch, "cuda"):
    torch.cuda.is_current_stream_capturing = lambda: False
    if hasattr(torch.cuda, "graphs"):
        torch.cuda.graphs.is_current_stream_capturing = lambda: False

try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_cuda_stream_capturing = lambda: False
except Exception:
    pass

# =====================================================
# Force UTF-8 stdout so debug prints (🌡️, °C, etc.) show
# correctly in PowerShell instead of as mojibake
# (ðŸŒ¡, Â°C). JSON responses to the frontend were already
# UTF-8 and unaffected by this — this only fixes what
# you see in the terminal.
# =====================================================

if hasattr(sys.stdout, "reconfigure"):

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel
import requests

from backend.agent.orchestrator import orchestrate
from backend.tools.weather_tool import (
    get_weather,
    get_weather_forecast,
    search_locations,
    reverse_geocode
)
from backend.tools.historical_tool import compare_weather
from backend.services.risk_engine import calculate_risk
from backend.services.impact_advisory import generate_advisory
from backend.services.nwp_service import get_gfs_forecast, get_wrf_info
from backend.tools.gru_tool import predict_latest_temperature
from backend.routes.alerts import router as alerts_router
from backend.routes.feedback import router as feedback_router


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="WeatherGPT+",
    description=(
        "Conversational AI & GIS Platform for weather forecasting, "
        "risk analysis, extreme alerts and numerical weather prediction"
    ),
    version="2.0.0",
    default_response_class=UTF8JSONResponse
)


# =====================================================
# CORS
# =====================================================

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://weather-gpt-production-4837.up.railway.app",
    "null"
]

extra_origins = os.getenv("CORS_ORIGINS", "")
if extra_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https://.*\.railway\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================================
# MOUNT ROUTERS
# =====================================================

app.include_router(alerts_router)
app.include_router(feedback_router)


# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    location_context: Optional[dict] = None
    conversation_history: Optional[list] = None



# =====================================================
# ROOT & HEALTH (MUST PRESERVE FOR TESTS)
# =====================================================

@app.get("/")
def root():
    return {
        "message": "WeatherGPT+ API is running",
        "status": "success",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =====================================================
# FRONTEND APP SERVING
# =====================================================

@app.get("/app")
def serve_app():
    return FileResponse("frontend/index.html", media_type="text/html; charset=utf-8")


@app.get("/frontend")
def serve_frontend():
    return FileResponse("frontend/index.html", media_type="text/html; charset=utf-8")


# =====================================================
# MAP TILE PROXY (SECURE OPENWEATHER TILES)
# =====================================================

VALID_TILE_LAYERS = {
    "precipitation": "precipitation_new",
    "precipitation_new": "precipitation_new",
    "temp": "temp_new",
    "temp_new": "temp_new",
    "wind": "wind_new",
    "wind_new": "wind_new",
    "clouds": "clouds_new",
    "clouds_new": "clouds_new",
    "pressure": "pressure_new",
    "pressure_new": "pressure_new"
}


@app.get("/api/map/tile/{layer}/{z}/{x}/{y}.png")
def map_tile(layer: str, z: int, x: int, y: int):
    """
    Proxies OpenWeather meteorological tile layers to Leaflet maps securely
    without exposing the backend OPENWEATHER_API_KEY to client JavaScript.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return Response(status_code=404, content=b"", media_type="image/png")

    actual_layer = VALID_TILE_LAYERS.get(layer.lower())
    if not actual_layer:
        return Response(status_code=400, content=b"", media_type="image/png")

    url = f"https://tile.openweathermap.org/map/{actual_layer}/{z}/{x}/{y}.png?appid={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=1800"}
        )
    except Exception:
        return Response(status_code=502, content=b"", media_type="image/png")


# =====================================================
# UNIFIED WEATHER & GIS INTELLIGENCE ENDPOINT
# =====================================================

@app.get("/api/weather")
def unified_weather(
    city: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None
):
    """
    Returns full meteorological intelligence: real-time observation, 5-day daily forecast,
    chronological 3-hour progression, 4-tier risk calculation, impact advisory,
    historical climatology comparison, WeatherGRU prediction, and NOAA GFS NWP.
    """
    try:
        current = get_weather(city=city, lat=lat, lon=lon)
        eff_city = current.get("city") or city or "Kurnool"
        eff_lat = current.get("lat") or lat
        eff_lon = current.get("lon") or lon

        # 5-Day Forecast
        forecast_data = get_weather_forecast(city=eff_city, lat=eff_lat, lon=eff_lon, days_ahead=None)

        # Historical comparison against 30-year climatological normal
        historical = compare_weather(eff_city, current)

        # 4-Tier Risk Assessment & Actionable Advisory
        risk = calculate_risk(current, historical)
        advisory = generate_advisory(current, risk)

        # WeatherGRU ML Next-Hour Prediction
        gru_pred = None
        try:
            raw_gru = predict_latest_temperature()
            gru_pred = {
                "predicted_temperature_celsius": raw_gru.get("prediction_celsius") if isinstance(raw_gru, dict) else float(raw_gru),
                "model": "WeatherGRU (2-Layer Recurrent Unit)",
                "confidence_label": "Model confidence unavailable",
                "confidence_note": "Deterministic GRU regression without calibration ensemble",
                "source": "ERA5 Land Reanalysis (era5_merged.nc)",
                "horizon": "next_hour"
            }
        except Exception as e:
            gru_pred = {
                "error": str(e),
                "predicted_temperature_celsius": None,
                "confidence_label": "Unavailable"
            }

        # NOAA GFS Numerical Weather Prediction
        gfs_data = None
        if eff_lat is not None and eff_lon is not None:
            try:
                gfs_data = get_gfs_forecast(eff_lat, eff_lon, days=3)
            except Exception as e:
                gfs_data = {"error": str(e), "success": False}

        return {
            "success": True,
            "location": {
                "city": eff_city,
                "name_te": current.get("name_te", eff_city),
                "country": current.get("country", "IN"),
                "lat": eff_lat,
                "lon": eff_lon
            },
            "current": current,
            "forecast_daily": forecast_data.get("daily", []),
            "forecast_hourly": forecast_data.get("hourly", []),
            "risk": risk,
            "advisory": advisory,
            "historical": historical,
            "gru_prediction": gru_pred,
            "nwp_gfs": gfs_data,
            "sources": [
                {"name": "OpenWeather API", "role": "Real-Time Weather & 5-Day Forecast"},
                {"name": "WeatherGRU Model", "role": "Next-Hour Deep Learning Temperature Regression"},
                {"name": "NOAA GFS (0.25°)", "role": "Numerical Weather Prediction Model"},
                {"name": "IMD Bulletins & FAISS RAG", "role": "Official Indian Meteorological Knowledge"}
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return UTF8JSONResponse(
            status_code=502 if "OPENWEATHER" in str(e) or "Weather" in str(e) else 400,
            content={
                "success": False,
                "error": str(e)
            }
        )


# =====================================================
# GEOCODING ENDPOINTS
# =====================================================

@app.get("/api/location/search")
def location_search(q: str = Query(...), limit: int = 5):
    return {"results": search_locations(q, limit=limit)}


@app.get("/api/location/reverse")
def location_reverse(lat: float, lon: float):
    return reverse_geocode(lat, lon)


# =====================================================
# NUMERICAL WEATHER PREDICTION ENDPOINTS
# =====================================================

@app.get("/api/nwp/gfs")
def nwp_gfs(lat: float, lon: float, days: int = 3):
    return get_gfs_forecast(lat, lon, days=days)


@app.get("/api/nwp/wrf")
def nwp_wrf():
    return get_wrf_info()


# =====================================================
# BACKWARD-COMPATIBLE /weather/{city} ENDPOINT
# =====================================================

@app.get("/weather/{city}")
def weather(city: str):
    try:
        return get_weather(city=city)
    except Exception as e:
        return UTF8JSONResponse(
            status_code=502,
            content={
                "success": False,
                "tool": "weather",
                "type": "weather_error",
                "error": str(e)
            }
        )


# =====================================================
# CHAT ENDPOINT (WITH LOCATION CONTEXT & SEMANTIC INTENT)
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = orchestrate(
            request.message,
            language=request.language,
            location_context=request.location_context,
            conversation_history=request.conversation_history
        )
        return result
    except Exception as e:
        return UTF8JSONResponse(
            status_code=500,
            content={
                "success": False,
                "tool": "unknown",
                "type": "server_error",
                "message": request.message,
                "language": request.language,
                "error": str(e),
                "answer": f"⚠️ Server error processing request: {str(e)}"
            }
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
