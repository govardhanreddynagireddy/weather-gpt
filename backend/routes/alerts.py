from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional

from backend.tools.weather_tool import get_weather
from backend.tools.historical_tool import compare_weather
from backend.services.risk_engine import calculate_risk
from backend.services.impact_advisory import generate_advisory

router = APIRouter(prefix="/alerts", tags=["alerts"])

def _generate_alerts_payload(city: str):
    if not city or city.strip().lower() == "undefined":
        raise ValueError("Invalid city specified.")

    city = city.strip()
    weather = get_weather(city)
    historical = compare_weather(city, weather)
    risk = calculate_risk(weather, historical)
    advisory = generate_advisory(weather, risk)

    alerts_list = []
    if risk["risk_level"] in ["HIGH", "MEDIUM"]:
        for reason in risk.get("reasons", []):
            alerts_list.append({
                "severity": risk["risk_level"],
                "title": f"Weather Alert: {reason}",
                "message": (
                    advisory["possible_impacts"][0]
                    if advisory.get("possible_impacts")
                    else f"Caution advised due to {reason.lower()}."
                ),
                "recommendations": advisory.get("recommendations", []),
                "source": "WeatherGPT Risk Engine",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    return {
        "location": weather.get("city", city),
        "has_alert": len(alerts_list) > 0,
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "reasons": risk.get("reasons", []),
        "alerts": alerts_list,
        "advisory": advisory
    }

@router.get("")
def get_alerts_query(city: str = Query(default="Kurnool")):
    try:
        return _generate_alerts_payload(city)
    except Exception as e:
        return JSONResponse(
            status_code=502 if "OPENWEATHER" in str(e) or "Weather API" in str(e) else 400,
            content={
                "success": False,
                "location": city,
                "has_alert": False,
                "error": str(e),
                "alerts": []
            }
        )

@router.get("/{city}")
def get_alerts_path(city: str):
    try:
        return _generate_alerts_payload(city)
    except Exception as e:
        return JSONResponse(
            status_code=502 if "OPENWEATHER" in str(e) or "Weather API" in str(e) else 400,
            content={
                "success": False,
                "location": city,
                "has_alert": False,
                "error": str(e),
                "alerts": []
            }
        )
