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


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

    try:

        return get_weather(city)

    except Exception as e:

        return JSONResponse(

            status_code=502,

            content={

                "success":False,

                "tool":"weather",

                "type":"weather_error",

                "error":str(e)

            }

        )


# =====================================================
# CHAT
# =====================================================

@app.post("/chat")
def chat(request:ChatRequest):

    # =================================================
    # orchestrate() already converts per-tool failures
    # (weather / gru / rag) into JSON error dicts. This
    # outer try/except is a last-resort safety net so
    # that ANY unexpected exception in routing itself
    # still returns JSON instead of a raw 500 / dropped
    # connection.
    # =================================================

    try:

        result=orchestrate(

            request.message
        )

        return result

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={

                "success":False,

                "tool":"unknown",

                "type":"server_error",

                "message":request.message,

                "error":str(e),

                "answer":f"⚠️ Server error processing request: {str(e)}"

            }

        )
