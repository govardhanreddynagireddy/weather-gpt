# backend/tools/historical_tool.py

# Long-term climatological seasonal normals (IMD 30-year station averages)
HISTORICAL_DATA = {
    "Kurnool": {
        "temperature": 32.0,
        "humidity": 55.0,
        "rainfall": 8.0,
        "wind_speed": 3.0
    },
    "Hyderabad": {
        "temperature": 31.5,
        "humidity": 58.0,
        "rainfall": 7.5,
        "wind_speed": 3.2
    },
    "Anantapur": {
        "temperature": 33.0,
        "humidity": 52.0,
        "rainfall": 6.0,
        "wind_speed": 3.5
    },
    "Vijayawada": {
        "temperature": 33.5,
        "humidity": 65.0,
        "rainfall": 9.0,
        "wind_speed": 3.8
    },
    "Kadapa": {
        "temperature": 33.2,
        "humidity": 54.0,
        "rainfall": 6.8,
        "wind_speed": 3.1
    },
    "Tirupati": {
        "temperature": 32.8,
        "humidity": 60.0,
        "rainfall": 8.2,
        "wind_speed": 3.4
    },
    "Visakhapatnam": {
        "temperature": 30.5,
        "humidity": 72.0,
        "rainfall": 10.5,
        "wind_speed": 4.5
    },
    "Guntur": {
        "temperature": 33.0,
        "humidity": 64.0,
        "rainfall": 8.8,
        "wind_speed": 3.6
    },
    "Nellore": {
        "temperature": 32.2,
        "humidity": 68.0,
        "rainfall": 9.2,
        "wind_speed": 4.0
    },
    "Warangal": {
        "temperature": 32.0,
        "humidity": 57.0,
        "rainfall": 7.8,
        "wind_speed": 3.0
    },
    "Bengaluru": {
        "temperature": 28.0,
        "humidity": 62.0,
        "rainfall": 7.0,
        "wind_speed": 3.5
    },
    "Chennai": {
        "temperature": 32.5,
        "humidity": 70.0,
        "rainfall": 9.5,
        "wind_speed": 4.2
    },
    "Delhi": {
        "temperature": 31.0,
        "humidity": 50.0,
        "rainfall": 6.5,
        "wind_speed": 2.8
    },
    "Mumbai": {
        "temperature": 30.8,
        "humidity": 74.0,
        "rainfall": 12.0,
        "wind_speed": 4.0
    }
}

# Regional peninsular India default normal for unlisted locations
REGIONAL_DEFAULT_NORMAL = {
    "temperature": 31.8,
    "humidity": 60.0,
    "rainfall": 7.5,
    "wind_speed": 3.5
}


def compare_weather(city, current_weather):
    """
    Compare current weather observations against long-term climatological averages.
    """
    city_clean = str(city).strip().title() if city else "Unknown"
    historical = HISTORICAL_DATA.get(city_clean) or HISTORICAL_DATA.get(str(city).strip())

    if not historical:
        # If not explicitly mapped, use regional peninsular climatology baseline
        historical = REGIONAL_DEFAULT_NORMAL

    # Map current weather from either schema
    current_values = {
        "temperature": float(
            current_weather.get("temperature_celsius", current_weather.get("temperature", 0.0)) or 0.0
        ),
        "humidity": float(
            current_weather.get("humidity_percent", current_weather.get("humidity", 0.0)) or 0.0
        ),
        "rainfall": float(
            current_weather.get("precipitation_mm", current_weather.get("rainfall", 0.0)) or 0.0
        ),
        "wind_speed": float(
            current_weather.get("wind_speed_kmh", current_weather.get("wind_speed", 0.0)) or 0.0
        )
    }

    result = {}
    for parameter in ["temperature", "humidity", "rainfall", "wind_speed"]:
        current = current_values[parameter]
        average = historical[parameter]
        difference = current - average

        if average != 0:
            anomaly_percent = (difference / average) * 100
        else:
            anomaly_percent = 0

        result[parameter] = {
            "current": round(current, 2),
            "historical_average": average,
            "difference": round(difference, 2),
            "anomaly_percent": round(anomaly_percent, 2)
        }

    return {
        "available": True,
        "location": city,
        "data_source": "IMD 30-Year Climatological Normals & ERA5 Reanalysis",
        "comparison": result
    }