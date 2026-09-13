from fastapi import FastAPI

app=FastAPI(
    title="WeatherGPT+",
    description="Conversational AI for weather forecasting, risk analysis and alerts",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message":"WeatherGPT+ API is running",
        "status":"success"
    }


@app.get("/health")
def health():
    return {
        "status":"healthy"
    }