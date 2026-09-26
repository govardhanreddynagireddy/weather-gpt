"""
NWP (Numerical Weather Prediction) Service
Integrates real NOAA Global Forecast System (GFS 0.25°) model forecasts and
defines an honest extension point for high-resolution WRF mesoscale modeling.
"""

from abc import ABC, abstractmethod
import requests
from typing import Dict, Any, Optional
from datetime import datetime


class NWPProvider(ABC):
    @abstractmethod
    def get_forecast(self, lat: float, lon: float, days: int = 3) -> Dict[str, Any]:
        """Fetch numerical weather model forecast for given coordinates."""
        pass


class GFSProvider(NWPProvider):
    """
    NOAA Global Forecast System (GFS 0.25° resolution) Provider.
    Queries real-time GFS model output via Open-Meteo GFS integration.
    Zero API key required; verified real meteorological physics simulation.
    """

    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.model_name = "gfs_seamless"
        self.agency = "National Oceanic and Atmospheric Administration (NOAA / NCEP)"
        self.grid_resolution = "0.25° (~28 km)"

    def get_forecast(self, lat: float, lon: float, days: int = 3) -> Dict[str, Any]:
        params = {
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "models": self.model_name,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure,cloud_cover",
            "forecast_days": min(max(days, 1), 7),
            "timezone": "auto"
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly") or {}
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            rain = hourly.get("precipitation", [])
            wind = hourly.get("wind_speed_10m", [])
            humidity = hourly.get("relative_humidity_2m", [])
            pressure = hourly.get("surface_pressure", [])
            clouds = hourly.get("cloud_cover", [])

            forecast_entries = []
            for i in range(min(len(times), 48)):  # 48 hours
                t_str = times[i]
                forecast_entries.append({
                    "time": t_str,
                    "temperature_celsius": round(float(temps[i]), 1) if i < len(temps) and temps[i] is not None else None,
                    "precipitation_mm": round(float(rain[i]), 2) if i < len(rain) and rain[i] is not None else 0.0,
                    "wind_speed_kmh": round(float(wind[i]), 1) if i < len(wind) and wind[i] is not None else None,
                    "humidity_percent": int(humidity[i]) if i < len(humidity) and humidity[i] is not None else None,
                    "pressure_hpa": round(float(pressure[i]), 1) if i < len(pressure) and pressure[i] is not None else None,
                    "cloud_cover_percent": int(clouds[i]) if i < len(clouds) and clouds[i] is not None else None
                })

            return {
                "success": True,
                "model": "NOAA GFS (Global Forecast System)",
                "model_identifier": self.model_name,
                "agency": self.agency,
                "grid_resolution": self.grid_resolution,
                "latitude": lat,
                "longitude": lon,
                "forecast_hours_count": len(forecast_entries),
                "hourly": forecast_entries,
                "data_source": "NOAA / Open-Meteo GFS API",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "model": "NOAA GFS",
                "error": str(e),
                "latitude": lat,
                "longitude": lon,
                "hourly": []
            }


class WRFProvider(NWPProvider):
    """
    Weather Research and Forecasting (WRF) Mesoscale Model Extension Point.
    WRF requires High-Performance Computing (HPC) nodes running WRF Preprocessing
    System (WPS: geogrid, ungrib, metgrid) and numerical solver (real.exe, wrf.exe)
    initialized from GFS boundary conditions.
    """

    def __init__(self):
        self.model_name = "WRF-ARW (Advanced Research WRF)"
        self.status = "Extension Point / HPC Required"

    def get_forecast(self, lat: float, lon: float, days: int = 3) -> Dict[str, Any]:
        return {
            "success": False,
            "model": self.model_name,
            "status": self.status,
            "is_active": False,
            "note": (
                "Local WRF execution is architected as an extension point. Full mesoscale numerical "
                "integration requires high-performance cluster computing with boundary data initialized "
                "from NOAA GFS."
            ),
            "recommended_provider": "GFSProvider"
        }


# Global singleton provider instance
_gfs_provider = GFSProvider()
_wrf_provider = WRFProvider()


def get_gfs_forecast(lat: float, lon: float, days: int = 3) -> Dict[str, Any]:
    """Retrieve real GFS numerical weather prediction for coordinates."""
    return _gfs_provider.get_forecast(lat, lon, days=days)


def get_wrf_info() -> Dict[str, Any]:
    """Retrieve WRF extension point metadata."""
    return _wrf_provider.get_forecast(0.0, 0.0)
