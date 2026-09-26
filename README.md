# 🌤️ WeatherGPT+

**WeatherGPT+** is an authoritative, interactive **GIS Weather Intelligence & Conversational AI Platform**. It unites real-time meteorological observations, deep learning temperature prediction (PyTorch WeatherGRU on ERA5 reanalysis), physics-based Numerical Weather Prediction (NOAA Global Forecast System GFS 0.25°), official IMD bulletin knowledge retrieval (FAISS + sentence-transformers), a 4-tier risk engine, agricultural decision support, and Google Gemini NLG in both **English** and **Telugu**, coupled with an interactive Leaflet GIS workstation and browser-native voice assistance.

---

## 🏗️ Architecture & Grounding Flow

WeatherGPT+ adheres strictly to the rule: **Real Meteorological Data $\rightarrow$ Risk & Advisory Analysis $\rightarrow$ Structured Context $\rightarrow$ AI Explanation $\rightarrow$ GIS UI**. Gemini never guesses numerical weather values or fabricates emergency warnings.

```
                      USER (Map Click / Voice / Text in EN or TE)
                                           │
                                           ▼
                            Semantic Entity & Intent Parser
                    (Extracts time horizon, location & UI actions)
                                           │
                                           ▼
                                 FASTAPI WORKSTATION
                     (/api/weather, /chat, /api/map/tile, /api/nwp)
                                           │
                                           ▼
                                     ORCHESTRATOR
                                           │
      ┌──────────────────┬─────────────────┼──────────────────┬──────────────────┐
      │                  │                 │                  │                  │
      ▼                  ▼                 ▼                  ▼                  ▼
OpenWeather API     WeatherGRU (ML)  NOAA GFS (0.25°)    FAISS / IMD RAG    Climatology
(Observed & 5-Day)  (Next-Hour Temp) (NWP Physics Model) (Official Bulletins) (30-Yr Normals)
      │                  │                 │                  │                  │
      └──────────────────┴─────────────────┼──────────────────┴──────────────────┘
                                           │
                                           ▼
                             4-Tier Risk Engine & Advisory
                        (LOW, MEDIUM, HIGH, SEVERE Categories)
                       (Farmer Advisory + Public Safety Guidance)
                                           │
                                           ▼
                               STRUCTURED TRUSTED CONTEXT
                                           │
                                           ▼
                                 GEMINI NLG EXPLAINER
                             (English 'en' / Telugu 'te')
                            (Deterministic fallback safety)
                                           │
                                           ▼
                            INTERACTIVE GIS WORKSTATION
            (Leaflet Map + Tile Overlays + Chart.js + Synchronized Chat)
```

---

## ✨ Features

- **Interactive Leaflet GIS Workstation**:
  - Click anywhere on the map to inspect weather, reverse geocode, and update the entire dashboard and chat context.
  - Real meteorological tile layer overlays served through a secure backend proxy:
    - 🌧️ Precipitation / Rain
    - 🌡️ Temperature
    - 💨 Wind Speed
    - ☁️ Cloud Cover
    - ⚠️ Dynamic Risk Zones
  - Draggable markers, zoom/pan controls, and visual weather pin popups.
- **Real-Time Weather & 5-Day Forecast**:
  - Full meteorological parameters: Temperature, Feels-like, Min/Max, Humidity, Pressure, Wind Speed & Direction, Visibility, Cloud Cover, Rainfall, Sunrise, and Sunset.
  - 5-Day daily summary cards + 48-hour chronological 3-hour forecast intervals.
  - Interactive Chart.js graph plotting temperature and precipitation progression.
- **Deep Learning Temperature Prediction (WeatherGRU)**:
  - 2-layer Recurrent Unit trained on ERA5 atmospheric reanalysis (`d2m`, `t2m`, `sp`, `tp`, `ssrd`, `skt`, `u10`, `v10`).
  - Next-hour temperature regression with honest confidence labeling (`"Model confidence unavailable"`).
  - Visual observation-to-prediction sequence chart.
- **Numerical Weather Prediction (NOAA GFS & WRF)**:
  - Integrated with real NOAA Global Forecast System (GFS 0.25° grid) numerical weather model.
  - WRF (Weather Research & Forecasting) mesoscale modeling extension point cleanly documented.
- **4-Tier Risk Engine (LOW, MEDIUM, HIGH, SEVERE)**:
  - Multi-variable risk assessment evaluating heat waves, torrential precipitation, storm-force winds, humidity, and historical anomalies.
- **Actionable Decision Support (Agricultural & Public)**:
  - 🚜 **Farmer Advisory**: Irrigation management, crop protection, pesticide spraying suitability, and livestock care.
  - 👥 **Public Safety**: Hydration, outdoor exposure limits, travel safety, and rain gear.
  - Distinct provenance tagging: *"WeatherGPT Advisory"* vs *"Official IMD Warning"*.
- **Official IMD Bulletin RAG**:
  - FAISS vector search over 396 official India Meteorological Department bulletins using `all-MiniLM-L6-v2`.
- **Historical Climatological Baseline**:
  - Compares current observations against 30-year IMD/ERA5 station averages for Indian regions.
- **Natural Language & Semantic Understanding (English & Telugu)**:
  - Understands colloquial Telugu phrases (*"రేపు వాతావరణం ఎలా ఉంటుంది?"*, *"repu weather ela untadhi"*, *"వర్షం వస్తుందా?"*).
  - Interprets relative time (*"tomorrow"*, *"repu"*) and location (*"here"*, *"near me"*) semantically without treating words like *"repu"* as cities.
- **Chat-to-Map Synchronization**:
  - Commands in chat (*"show rainfall"*, *"show wind"*, *"show risk"*, *"weather here"*) automatically trigger map layer changes and geolocation.
  - Interactive action suggestion chips (`[📍 View on Map]`, `[🌧️ Rain Layer]`, `[📊 Hourly Chart]`, `[⚠️ Advisory]`, `[📄 IMD Bulletins]`).
- **Browser-Native Voice Assistant**:
  - Speech-to-Text (STT) input with animated listening pulse in English (`en-IN`) and Telugu (`te-IN`).
  - Speech synthesis (TTS) reading out responses aloud with toggle.

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
| :--- | :--- | :--- |
| `OPENWEATHER_API_KEY` | Yes | OpenWeatherMap API key (current observations, 5-day forecasts, geocoding & tiles). |
| `GEMINI_API_KEY` | Recommended | Google Gemini API key for natural language explanation layer. |
| `GEMINI_MODEL` | No | Model name (default: `gemini-3.6-flash` or `gemini-2.5-flash`). |
| `PORT` | No | Server port (default: `8000`, set automatically on Render/Railway). |

> [!CAUTION]
> Never commit `.env` or hardcode API keys in the repository.

---

## 🚀 Local Setup & Installation

Managed with [uv](https://github.com/astral-sh/uv).

### 1. Install Dependencies
```bash
cd weather-gpt
uv sync
```

### 2. Start the Application
```bash
# Run FastAPI server
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
- **Interactive GIS Workstation**: [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) (or open `frontend/index.html`)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API root check (`{"status": "success"}`). |
| `/health` | `GET` | Health status check (`{"status": "healthy"}`). |
| `/app` | `GET` | Serves the interactive GIS workstation frontend. |
| `/api/weather` | `GET` | Unified weather payload (current, 5-day daily, hourly, risk, advisory, GRU, GFS) by `city` or `lat`/`lon`. |
| `/weather/{city}` | `GET` | Current weather observation for a city. |
| `/api/location/search` | `GET` | Direct geocoding location search with Telugu transliterations. |
| `/api/location/reverse` | `GET` | Reverse geocodes latitude/longitude coordinates into place names. |
| `/api/map/tile/{layer}/{z}/{x}/{y}.png` | `GET` | Secure proxy for OpenWeather map tiles (rain, temp, wind, clouds). |
| `/api/nwp/gfs` | `GET` | Real NOAA GFS (0.25°) numerical weather prediction hourly forecast. |
| `/api/nwp/wrf` | `GET` | WRF mesoscale modeling extension point metadata. |
| `/alerts/{city}` | `GET` | Active meteorological alerts and advisories. |
| `/chat` | `POST` | Conversational endpoint with semantic parsing, location context, and chat-map sync. |
| `/feedback` | `POST` | User rating and feedback submission. |

---

## 🌐 Render / Railway Cloud Deployment

### Deployment Settings
- **Build Command**: `uv sync` (or `pip install -r backend/requirements.txt`)
- **Start Command**: `uv run uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Port Binding**: Automatically reads `$PORT` from environment.
- **Hardware Requirement**: Runs completely on CPU (GPU is not required).

---

## 🧪 Testing

Execute the comprehensive automated test suite:

```bash
uv run python -m backend.test_suite
```

Execute deep verification across all tools:

```bash
uv run python -m backend.verify_real_calls
```

Execute historical regression test:

```bash
uv run python backend/test_weather.py
```
