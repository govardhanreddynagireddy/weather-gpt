def calculate_risk(weather,historical):

    score=0
    reasons=[]

    temperature=weather["temperature"]
    rainfall=weather["rainfall"]
    wind_speed=weather["wind_speed"]
    humidity=weather["humidity"]

    # Temperature risk
    if temperature>=40:
        score+=30
        reasons.append("Very high temperature")
    elif temperature>=35:
        score+=20
        reasons.append("High temperature")

    # Rainfall risk
    if rainfall>=100:
        score+=40
        reasons.append("Very heavy rainfall")
    elif rainfall>=50:
        score+=30
        reasons.append("Heavy rainfall")
    elif rainfall>=20:
        score+=15
        reasons.append("Moderate rainfall")

    # Wind risk
    if wind_speed>=15:
        score+=30
        reasons.append("Very strong winds")
    elif wind_speed>=10:
        score+=20
        reasons.append("Strong winds")
    elif wind_speed>=5:
        score+=10
        reasons.append("Moderate winds")

    # Humidity risk
    if humidity>=90:
        score+=10
        reasons.append("Very high humidity")

    # Historical anomaly
    if historical["available"]:

        comparison=historical["comparison"]

        temperature_anomaly=comparison["temperature"]["anomaly_percent"]

        if temperature_anomaly>=20:
            score+=15
            reasons.append("Temperature is significantly above historical levels")

        wind_anomaly=comparison["wind_speed"]["anomaly_percent"]

        if wind_anomaly>=30:
            score+=10
            reasons.append("Wind speed is significantly above historical levels")

    # Limit score
    score=min(score,100)

    # Risk level
    if score>=60:
        risk_level="HIGH"
    elif score>=30:
        risk_level="MEDIUM"
    else:
        risk_level="LOW"

    return {
        "risk_score":score,
        "risk_level":risk_level,
        "reasons":reasons
    }