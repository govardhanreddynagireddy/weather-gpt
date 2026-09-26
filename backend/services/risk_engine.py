def calculate_risk(weather, historical=None):
    """
    Evaluates meteorological risk based on real weather observations and historical anomalies.
    Returns risk score (0-100), 4-tier level (LOW, MEDIUM, HIGH, SEVERE), and contributing reasons.
    """
    score = 0
    reasons = []

    if not weather:
        return {
            "risk_score": 0,
            "risk_level": "LOW",
            "risk_tier": "LOW",
            "reasons": ["No weather data available"],
            "severity_badge": "🟢 LOW"
        }

    temperature = float(weather.get("temperature_celsius", weather.get("temperature", 0.0)) or 0.0)
    rainfall = float(weather.get("precipitation_mm", weather.get("rainfall", 0.0)) or 0.0)
    wind_speed = float(weather.get("wind_speed_kmh", weather.get("wind_speed", 0.0)) or 0.0)
    humidity = float(weather.get("humidity_percent", weather.get("humidity", 0.0)) or 0.0)

    # Temperature hazards
    if temperature >= 42:
        score += 35
        reasons.append("Extreme heat wave conditions")
    elif temperature >= 38:
        score += 25
        reasons.append("High heat stress")
    elif temperature >= 35:
        score += 15
        reasons.append("Elevated temperature")
    elif temperature <= 5 and temperature > 0:
        score += 20
        reasons.append("Cold wave conditions")

    # Rainfall / Flood hazards
    if rainfall >= 100:
        score += 45
        reasons.append("Extremely heavy rainfall (flood hazard)")
    elif rainfall >= 50:
        score += 30
        reasons.append("Heavy rainfall (waterlogging risk)")
    elif rainfall >= 20:
        score += 15
        reasons.append("Moderate rainfall")

    # Wind hazards
    if wind_speed >= 35:
        score += 35
        reasons.append("Gale / Storm-force winds")
    elif wind_speed >= 20:
        score += 25
        reasons.append("Strong gusty winds")
    elif wind_speed >= 10:
        score += 15
        reasons.append("Moderate winds")

    # Humidity / Heat Index interaction
    if humidity >= 90:
        score += 10
        reasons.append("Very high relative humidity")

    # Historical climatological anomaly
    if historical and historical.get("available") and "comparison" in historical:
        comparison = historical["comparison"]
        temp_comp = comparison.get("temperature") or {}
        wind_comp = comparison.get("wind_speed") or {}

        temp_anom = float(temp_comp.get("anomaly_percent", 0.0) or 0.0)
        if temp_anom >= 20:
            score += 15
            reasons.append("Temperature significantly above historical normal")

        wind_anom = float(wind_comp.get("anomaly_percent", 0.0) or 0.0)
        if wind_anom >= 30:
            score += 10
            reasons.append("Wind speed significantly above historical normal")

    # Cap score at 100
    score = min(score, 100)

    # 4-Tier official risk categories
    if score >= 85:
        risk_level = "SEVERE"
        severity_badge = "🔴 SEVERE"
    elif score >= 60:
        risk_level = "HIGH"
        severity_badge = "🟠 HIGH"
    elif score >= 30:
        risk_level = "MEDIUM"
        severity_badge = "🟡 MEDIUM"
    else:
        risk_level = "LOW"
        severity_badge = "🟢 LOW"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "risk_tier": risk_level,
        "severity_badge": severity_badge,
        "reasons": reasons if reasons else ["Normal weather conditions"]
    }