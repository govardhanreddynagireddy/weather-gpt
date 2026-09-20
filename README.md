# 🌤️ WeatherGPT+

**WeatherGPT+** is an intelligent, grounded conversational weather platform. It combines real-time meteorological observations, ML temperature predictions (PyTorch WeatherGRU trained on ERA5 reanalysis), official IMD bulletin retrieval (FAISS + sentence-transformers), a deterministic risk and impact advisory engine, and Google Gemini for natural language generation (NLG) in both **English** and **Telugu**, with browser-native voice interaction and Render-ready cloud deployment.

---

## 🏗️ Architecture & Grounding Flow

WeatherGPT+ treats Gemini strictly as an **explanation layer**, not as a factual data source. All facts, numbers, risk levels, and bulletins originate from authoritative backend components:

```
                      USER (Voice / Text)
                             │
                             ▼
                    FASTAPI (/chat, /alerts)
                             │
                             ▼
                        ORCHESTRATOR
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
 Weather API / Forecast  WeatherGRU (ERA5)   FAISS / IMD RAG
 (OpenWeatherMap)      (Next-Hour Temp)    (Bulletins/Warnings)
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                             ▼
                    RISK & ADVISORY ENGINE
                     (Deterministic Rules)
                             │
                             ▼
                 STRUCTURED TRUSTED CONTEXT
                             │
                             ▼
                 GEMINI NLG (Grounding Prompt)
                 (English 'en' / Telugu 'te')
                             │
                             ▼
                 NATURAL LANGUAGE RESPONSE
              (with Deterministic Fallback)
```

---

## ✨ Features

- **Real-Time Weather & 5-Day Forecasts**: Authoritative temperature, humidity, wind, and precipitation via OpenWeatherMap API.
- **Next-Hour ML Temperature Prediction**: Powered by a 2-layer GRU model (`WeatherGRU`) trained on ERA5 reanalysis data (`d2m`, `t2m`, `sp`, `tp`, `ssrd`, `skt`, `u10`, `v10`).
- **Official IMD Bulletin Grounding**: Semantic retrieval over indexed India Meteorological Department bulletins using `FAISS` and `all-MiniLM-L6-v2`.
- **Deterministic Risk Engine & Impact Advisory**: Calculates quantified risk scores (0–100) and risk levels (`LOW`, `MEDIUM`, `HIGH`) based on heat, heavy rainfall, high winds, and historical anomalies.
- **Proactive Alerts**: Active alert system exposed via `GET /alerts/{city}` and visible directly in the frontend dashboard.
- **Bilingual Support (English & Telugu)**:
  - English (`en`): Fluent, actionable explanations.
  - Telugu (`te`): Idiomatic conversational Telugu preserving exact numerical values and units.
- **Browser-Native Voice Interaction**:
  - Voice Input (Speech-to-Text): Web Speech API using `en-IN` and `te-IN`.
  - Voice Output (Text-to-Speech): Web Speech Synthesis reading out grounded responses.
- **Resilient Fallback**: If `GEMINI_API_KEY` is missing or the API encounters errors, the application seamlessly falls back to formatted deterministic responses without crashing.
- **Render-Ready Deployment**: Configured to dynamically bind to `0.0.0.0` and read `$PORT`.

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` in the root directory:

```bash
cp .env.example .env
```

| Variable | Required | Description |
| :--- | :--- | :--- |
| `OPENWEATHER_API_KEY` | Yes | API key for OpenWeatherMap (current observations & forecasts). |
| `GEMINI_API_KEY` | Recommended | Google Gemini API key for natural language generation. |
| `GEMINI_MODEL` | No | Model name (default: `gemini-2.5-flash`). |
| `PORT` | No | Server listening port (default: `8000`, set automatically on Render). |

> [!CAUTION]
> Never commit `.env` or hardcode API keys in the repository.

---

## 🚀 Local Setup & Installation

The project is managed with [uv](https://github.com/astral-sh/uv).

### 1. Install uv & Sync Dependencies

```bash
# Clone the repository
cd weather-gpt

# Sync virtual environment and dependencies
uv sync
```

### 2. Run the Backend

```bash
# Using uv
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Or directly with Python
python -m backend.main
```

The API will be accessible at:
- **API Root**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Current Weather**: `GET /weather/{city}`
- **Active Alerts**: `GET /alerts/{city}`
- **Chat Endpoint**: `POST /chat`

### 3. Run the Frontend

The frontend is a single standalone file at `frontend/index.html`. You can open it directly in a browser or serve it via any static server:

```bash
# Example: serve with Python
python -m http.server 3000 --directory frontend
```

Then visit [http://localhost:3000](http://localhost:3000).

---

## 🌐 Render Deployment

### 1. Web Service Settings
- **Environment**: Python 3.12+
- **Build Command**: `uv sync` (or `pip install -r backend/requirements.txt`)
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### 2. Environment Variables on Render
Configure the following in your Render dashboard:
- `OPENWEATHER_API_KEY`: Your OpenWeatherMap API key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: `gemini-2.5-flash`

---

## 🧪 Testing

Run the automated test suite:

```bash
uv run python -m backend.test_suite
```

Or run the historical comparison regression test:

```bash
uv run python backend/test_weather.py
```
