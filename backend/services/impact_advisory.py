def generate_advisory(weather,risk):

    risk_level=risk["risk_level"]
    reasons=risk["reasons"]

    impacts=[]
    recommendations=[]

    temperature=weather["temperature"]
    rainfall=weather["rainfall"]
    wind_speed=weather["wind_speed"]

    # Temperature
    if temperature>=40:
        impacts.append("Extreme heat conditions may cause heat-related health risks.")
        recommendations.append("Avoid prolonged outdoor exposure and stay hydrated.")

    elif temperature>=35:
        impacts.append("High temperatures may cause heat-related discomfort.")
        recommendations.append("Stay hydrated and avoid prolonged exposure during peak afternoon hours.")

    # Rainfall
    if rainfall>=100:
        impacts.append("Very heavy rainfall may cause flooding or waterlogging.")
        recommendations.append("Avoid low-lying areas and follow official warnings.")

    elif rainfall>=50:
        impacts.append("Heavy rainfall may cause localized waterlogging.")
        recommendations.append("Avoid vulnerable roads and monitor local warnings.")

    # Wind
    if wind_speed>=15:
        impacts.append("Very strong winds may damage weak structures and create hazards.")
        recommendations.append("Stay away from weak structures and unsecured objects.")

    elif wind_speed>=10:
        impacts.append("Strong winds may create localized hazards.")
        recommendations.append("Secure loose outdoor objects and monitor weather updates.")

    # Default
    if not impacts:
        impacts.append("No major immediate weather impact detected.")

    if not recommendations:
        recommendations.append("Continue monitoring weather conditions.")

    return {
        "risk_level":risk_level,
        "possible_impacts":impacts,
        "recommendations":recommendations,
        "reasons":reasons
    }