import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")


# =====================================================
# GEOCODING & REVERSE GEOCODING
# =====================================================

def search_locations(query, limit=5):
    """
    Search for locations matching a query string.
    Returns list of dicts with name, local_names, lat, lon, country, state.
    """
    if not query or not str(query).strip():
        return []

    api_key = os.getenv("OPENWEATHER_API_KEY") or API_KEY
    if not api_key:
        return []

    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": str(query).strip(),
        "limit": limit,
        "appid": api_key
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        output = []
        for r in results:
            local_names = r.get("local_names") or {}
            output.append({
                "name": r.get("name", ""),
                "name_te": local_names.get("te", r.get("name", "")),
                "name_hi": local_names.get("hi", r.get("name", "")),
                "lat": round(float(r.get("lat", 0.0)), 4),
                "lon": round(float(r.get("lon", 0.0)), 4),
                "country": r.get("country", ""),
                "state": r.get("state", "")
            })
        return output
    except Exception as e:
        print(f"Geocoding search failed for '{query}': {e}")
        return []


def reverse_geocode(lat, lon):
    """
    Reverse geocode coordinates into a city/place name and metadata.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY") or API_KEY
    if not api_key:
        return {"name": f"Location ({lat:.2f}, {lon:.2f})", "country": "IN"}

    url = "http://api.openweathermap.org/geo/1.0/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "limit": 1,
        "appid": api_key
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json()
        if items and len(items) > 0:
            first = items[0]
            local = first.get("local_names") or {}
            return {
                "name": first.get("name", ""),
                "name_te": local.get("te", first.get("name", "")),
                "country": first.get("country", "IN"),
                "state": first.get("state", "")
            }
    except Exception as e:
        print(f"Reverse geocode failed for ({lat}, {lon}): {e}")

    return {"name": f"{lat:.2f}, {lon:.2f}", "country": "IN"}


# =====================================================
# CURRENT WEATHER
# =====================================================

def get_weather(city=None, lat=None, lon=None):
    """
    Fetch current real-time weather from OpenWeatherMap.
    Supports either city string or (lat, lon) coordinates.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY") or API_KEY
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "appid": api_key,
        "units": "metric"
    }

    if lat is not None and lon is not None:
        try:
            params["lat"] = float(lat)
            params["lon"] = float(lon)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    elif city and str(city).strip().lower() != "undefined":
        params["q"] = str(city).strip()
    else:
        raise ValueError("Invalid city or coordinates")

    last_err = None
    data = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            last_err = e
            if attempt == 1:
                raise last_err

    if not data:
        raise ValueError("Empty response received from OpenWeather API")

    main = data.get("main") or {}
    wind = data.get("wind") or {}
    sys_data = data.get("sys") or {}
    coord = data.get("coord") or {}

    rain_data = data.get("rain") or {}
    precipitation = 0.0
    if isinstance(rain_data, dict):
        precipitation = rain_data.get("1h", rain_data.get("3h", 0.0)) or 0.0

    weather_desc = "Clear"
    weather_icon = "01d"
    weather_main = "Clear"
    if data.get("weather") and len(data["weather"]) > 0:
        weather_desc = data["weather"][0].get("description", "Clear")
        weather_icon = data["weather"][0].get("icon", "01d")
        weather_main = data["weather"][0].get("main", "Clear")

    tz_offset = data.get("timezone", 0)
    sunrise_ts = sys_data.get("sunrise")
    sunset_ts = sys_data.get("sunset")
    sunrise_str = (
        datetime.fromtimestamp(sunrise_ts, timezone(timedelta(seconds=tz_offset))).strftime("%H:%M")
        if sunrise_ts else "--"
    )
    sunset_str = (
        datetime.fromtimestamp(sunset_ts, timezone(timedelta(seconds=tz_offset))).strftime("%H:%M")
        if sunset_ts else "--"
    )

    resolved_city = data.get("name") or (str(city).strip() if city else "Unknown Location")
    lat_val = coord.get("lat", lat)
    lon_val = coord.get("lon", lon)

    # Localized Telugu name if available
    name_te = resolved_city
    if lat_val and lon_val:
        geo_info = reverse_geocode(lat_val, lon_val)
        if geo_info.get("name_te"):
            name_te = geo_info["name_te"]

    return {
        "city": resolved_city,
        "name_te": name_te,
        "country": sys_data.get("country", "IN"),
        "lat": round(float(lat_val), 4) if lat_val is not None else None,
        "lon": round(float(lon_val), 4) if lon_val is not None else None,
        "temperature_celsius": round(float(main.get("temp", 0.0)), 2),
        "feels_like_celsius": round(float(main.get("feels_like", main.get("temp", 0.0))), 2),
        "temp_min_celsius": round(float(main.get("temp_min", main.get("temp", 0.0))), 2),
        "temp_max_celsius": round(float(main.get("temp_max", main.get("temp", 0.0))), 2),
        "humidity_percent": int(main.get("humidity", 0)),
        "pressure_hpa": int(main.get("pressure", 1013)),
        "wind_speed_kmh": round(float(wind.get("speed", 0.0)) * 3.6, 2),
        "wind_direction_deg": int(wind.get("deg", 0)),
        "visibility_km": round(float(data.get("visibility", 10000)) / 1000.0, 1),
        "clouds_percent": int(data.get("clouds", {}).get("all", 0)),
        "weather_condition": weather_main,
        "weather_description": weather_desc,
        "weather_icon": weather_icon,
        "precipitation_mm": round(float(precipitation), 2),
        "sunrise": sunrise_str,
        "sunset": sunset_str,
        "time": datetime.now().strftime("%H:%M"),
        "timestamp": datetime.now().isoformat(),
        "data_source": "OpenWeather"
    }


# =====================================================
# WEATHER FORECAST (5-DAY / 3-HOURLY)
# =====================================================

def get_weather_forecast(city=None, lat=None, lon=None, days_ahead=1):
    """
    Fetch 5-day / 3-hour forecast from OpenWeatherMap.
    Supports either city string or (lat, lon) coordinates.
    Filters target date if days_ahead is set, and provides both daily & hourly summaries.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY") or API_KEY
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY is missing")

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "appid": api_key,
        "units": "metric"
    }

    if lat is not None and lon is not None:
        try:
            params["lat"] = float(lat)
            params["lon"] = float(lon)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    elif city and str(city).strip().lower() != "undefined":
        params["q"] = str(city).strip()
    else:
        raise ValueError("Invalid city or coordinates")

    last_err = None
    data = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            last_err = e
            if attempt == 1:
                raise last_err

    if not data:
        raise ValueError("Empty response received from OpenWeather Forecast API")

    resolved_city = data.get("city", {}).get("name") or (str(city).strip() if city else "Unknown Location")
    country = data.get("city", {}).get("country", "IN")
    coord = data.get("city", {}).get("coord") or {}

    target_date = (
        datetime.now() + timedelta(days=days_ahead)
    ).date() if days_ahead is not None else None

    target_forecasts = []
    all_hourly = []
    daily_groups = {}

    for item in data.get("list", []):
        dt_val = item.get("dt")
        forecast_time = datetime.fromtimestamp(dt_val) if dt_val else datetime.now()
        item_date = forecast_time.date()
        date_str = str(item_date)

        rain_data = item.get("rain") or {}
        rain_val = 0.0
        if isinstance(rain_data, dict):
            rain_val = rain_data.get("3h", 0.0) or 0.0

        w_desc = "Clear"
        w_icon = "01d"
        w_main = "Clear"
        if item.get("weather") and len(item["weather"]) > 0:
            w_desc = item["weather"][0].get("description", "Clear")
            w_icon = item["weather"][0].get("icon", "01d")
            w_main = item["weather"][0].get("main", "Clear")

        main_obj = item.get("main") or {}
        wind_obj = item.get("wind") or {}

        hourly_entry = {
            "dt": dt_val,
            "date": date_str,
            "time": forecast_time.strftime("%H:%M"),
            "datetime": forecast_time.strftime("%Y-%m-%d %H:%M"),
            "temperature": round(float(main_obj.get("temp", 0.0)), 2),
            "feels_like": round(float(main_obj.get("feels_like", 0.0)), 2),
            "temp_min": round(float(main_obj.get("temp_min", 0.0)), 2),
            "temp_max": round(float(main_obj.get("temp_max", 0.0)), 2),
            "humidity": int(main_obj.get("humidity", 0)),
            "pressure": int(main_obj.get("pressure", 1013)),
            "wind_speed": round(float(wind_obj.get("speed", 0.0)) * 3.6, 2),
            "wind_direction": int(wind_obj.get("deg", 0)),
            "weather": w_desc,
            "weather_condition": w_main,
            "weather_icon": w_icon,
            "rain_probability": round(float(item.get("pop", 0.0)) * 100, 1),
            "rainfall": round(float(rain_val), 2)
        }

        all_hourly.append(hourly_entry)

        if target_date is not None and item_date == target_date:
            target_forecasts.append(hourly_entry)

        if date_str not in daily_groups:
            daily_groups[date_str] = []
        daily_groups[date_str].append(hourly_entry)

    # Build structured 5-day daily forecast summary
    daily_summaries = []
    for d_str, entries in daily_groups.items():
        temps = [e["temperature"] for e in entries]
        rain_total = sum(e["rainfall"] for e in entries)
        max_pop = max(e["rain_probability"] for e in entries) if entries else 0.0

        # Choose midday condition as representative
        rep_entry = min(
            entries,
            key=lambda e: abs(int(e["time"].split(":")[0]) - 12) if ":" in e["time"] else 99
        )

        d_obj = datetime.strptime(d_str, "%Y-%m-%d")
        daily_summaries.append({
            "date": d_str,
            "day_name": d_obj.strftime("%a"),
            "max_temperature_celsius": round(max(temps), 1) if temps else "--",
            "min_temperature_celsius": round(min(temps), 1) if temps else "--",
            "precipitation_mm": round(rain_total, 2),
            "rain_probability": round(max_pop, 1),
            "weather_description": rep_entry["weather"],
            "weather_condition": rep_entry["weather_condition"],
            "weather_icon": rep_entry["weather_icon"]
        })

    # For backward compatibility with existing tests
    if days_ahead is not None and not target_forecasts:
        # If requested target date is slightly beyond 5-day limit, return closest available
        if all_hourly:
            target_forecasts = all_hourly[:8]
        else:
            raise ValueError("No forecast data available for the requested date.")

    return {
        "city": resolved_city,
        "country": country,
        "lat": coord.get("lat", lat),
        "lon": coord.get("lon", lon),
        "date": str(target_date) if target_date else (all_hourly[0]["date"] if all_hourly else None),
        "forecasts": target_forecasts,
        "daily": daily_summaries,
        "hourly": all_hourly[:16],  # Next 48 hours (3-hourly)
        "data_source": "OpenWeather 5-Day / 3-Hour Forecast"
    }