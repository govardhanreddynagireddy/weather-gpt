def generate_advisory(weather, risk):
    """
    Generates actionable decision support for farmers and the general public
    based on observed weather and calculated meteorological risk.
    """
    risk_level = risk.get("risk_level", "LOW") if isinstance(risk, dict) else "LOW"
    reasons = risk.get("reasons", []) if isinstance(risk, dict) else []

    impacts = []
    recommendations = []
    farmer_tips = []
    public_tips = []

    if not weather:
        return {
            "risk_level": risk_level,
            "possible_impacts": ["No weather data available."],
            "recommendations": ["Check current weather update."],
            "reasons": reasons,
            "farmer_advisory": {
                "irrigation": "Maintain standard schedule once weather update is received.",
                "crop_protection": "Monitor local conditions.",
                "spraying": "Check wind conditions before spraying.",
                "field_work": "Proceed with caution.",
                "recommendations": ["Check current weather updates."]
            },
            "public_advisory": {
                "hydration": "Stay hydrated.",
                "travel": "Normal precautions.",
                "outdoor_activity": "Normal outdoor activity.",
                "recommendations": ["Check current weather updates."]
            },
            "advisory_source": "WeatherGPT Advisory",
            "disclaimer": "WeatherGPT Advisory provides meteorological decision support and does not replace official IMD emergency bulletins."
        }

    temperature = float(weather.get("temperature_celsius", weather.get("temperature", 0.0)) or 0.0)
    rainfall = float(weather.get("precipitation_mm", weather.get("rainfall", 0.0)) or 0.0)
    wind_speed = float(weather.get("wind_speed_kmh", weather.get("wind_speed", 0.0)) or 0.0)
    humidity = float(weather.get("humidity_percent", weather.get("humidity", 0.0)) or 0.0)

    # 1. Temperature impacts & advisories
    irrigation_advice = "Standard irrigation schedule appropriate."
    crop_advice = "Normal vegetative growth conditions."

    if temperature >= 40:
        impacts.append("Extreme heat wave conditions: high risk of heatstroke, rapid soil moisture depletion, and crop wilting.")
        recommendations.append("Avoid prolonged outdoor exposure between 11 AM and 4 PM; consume ample fluids with electrolytes.")
        irrigation_advice = "Provide light, frequent irrigation during early morning or late evening hours to mitigate soil heat stress."
        crop_advice = "Apply mulch to retain root moisture; protect nursery beds and sensitive vegetable seedlings with shade nets."
        farmer_tips.append("Provide cool drinking water and shaded shelter for livestock; avoid heavy animal labor during midday.")
        public_tips.append("Stay indoors during peak sunlight; carry water, wear lightweight cotton clothing and a wide-brim hat.")
    elif temperature >= 35:
        impacts.append("High temperatures increase evapotranspiration rates and potential heat discomfort.")
        recommendations.append("Stay hydrated and schedule intensive outdoor labor during cooler morning or evening hours.")
        irrigation_advice = "Increase irrigation frequency slightly to compensate for daytime evaporation losses."
        crop_advice = "Monitor crops for midday wilting; ensure adequate soil moisture."
        farmer_tips.append("Ensure livestock enclosures are well-ventilated and shaded.")
        public_tips.append("Drink plenty of water even if not feeling thirsty.")
    elif temperature <= 10 and temperature > 0:
        impacts.append("Cold conditions may cause thermal shock to sensitive seedling crops and young livestock.")
        recommendations.append("Dress warmly and protect vulnerable individuals from cold exposure.")
        crop_advice = "Provide smoke/smudge covers or light evening irrigation to raise localized ground temperature."
        farmer_tips.append("Shelter livestock from cold drafts at night.")
        public_tips.append("Keep warm clothing on hand during early morning and late night.")

    # 2. Rainfall & Flood impacts & advisories
    if rainfall >= 100:
        impacts.append("Extremely heavy rainfall: severe waterlogging, flash flooding in low-lying fields, and potential road closures.")
        recommendations.append("Avoid traveling through inundated roads; move to elevated areas if living near low-lying drains.")
        irrigation_advice = "IMMEDIATELY SUSPEND all irrigation. Open field drainage channels to prevent root asphyxiation."
        crop_advice = "Postpone harvesting; secure harvested produce in watertight elevated storage."
        farmer_tips.append("Suspend all fertilizer and pesticide spraying — chemical runoff risk is critical.")
        public_tips.append("Do not attempt to cross flooded roadways or underpasses; keep emergency phone contacts accessible.")
    elif rainfall >= 50:
        impacts.append("Heavy rainfall: localized waterlogging, temporary drainage congestion, and slippery road conditions.")
        recommendations.append("Exercise caution on highways; avoid unpaved or waterlogged roads.")
        irrigation_advice = "Suspend irrigation. Ensure field bunds have overflow outlets."
        crop_advice = "Postpone pesticide spraying to avoid rain wash-off."
        farmer_tips.append("Check field drainage to prevent standing water around crop root zones.")
        public_tips.append("Carry sturdy rain protection and drive at reduced speeds with headlights on.")
    elif rainfall >= 10:
        impacts.append("Moderate rainfall provides beneficial soil moisture but may cause damp road conditions.")
        recommendations.append("Carry an umbrella or raincoat; monitor changing sky conditions.")
        irrigation_advice = "Delay scheduled irrigation; take advantage of natural precipitation."
        farmer_tips.append("Good conditions for rainfed sowing; postpone pesticide spray until foliage dries.")
        public_tips.append("Keep rain gear handy.")

    # 3. Wind impacts & advisories
    spraying_advice = "Wind conditions suitable for spraying."
    if wind_speed >= 35:
        impacts.append("Gale / Storm-force winds: hazard of falling tree branches, loose tin roofs, and lodging in tall crops (sugarcane, banana, maize).")
        recommendations.append("Stay indoors away from windows, unanchored hoardings, and power lines.")
        crop_advice = "Provide staking/propping for banana plants, papaya, and tall horticulture crops; tie crop bunches together."
        spraying_advice = "DO NOT SPRAY pesticides or foliar nutrients — excessive drift hazard."
        farmer_tips.append("Secure farm shed roofing sheets and tie down livestock shelter thatch.")
        public_tips.append("Park vehicles away from mature trees, electric poles, and construction sites.")
    elif wind_speed >= 20:
        impacts.append("Breezy to strong winds: potential spray drift and increased crop moisture loss.")
        recommendations.append("Secure loose outdoor objects and exercise care while cycling or two-wheeler driving.")
        spraying_advice = "Postpone pesticide spraying until wind speed subsides below 15 km/h to prevent chemical drift."
        farmer_tips.append("Check stakes on young orchard saplings.")
        public_tips.append("Be cautious on open elevated bridges and flyovers.")

    # 4. Defaults if clear weather
    if not impacts:
        impacts.append("Pleasant weather conditions; no immediate adverse meteorological impact detected.")

    if not recommendations:
        recommendations.append("Favorable conditions for routine outdoor activities, commuting, and standard farm management.")

    if not farmer_tips:
        farmer_tips.append("Optimal conditions for field preparation, weeding, fertilization, and regular harvest operations.")

    if not public_tips:
        public_tips.append("Ideal conditions for outdoor recreation, travel, and general daily commutes.")

    # Combine recommendations cleanly
    combined_recommendations = recommendations + farmer_tips[:1] + public_tips[:1]

    return {
        "risk_level": risk_level,
        "possible_impacts": impacts,
        "recommendations": combined_recommendations,
        "reasons": reasons,
        "farmer_advisory": {
            "irrigation": irrigation_advice,
            "crop_protection": crop_advice,
            "spraying": spraying_advice,
            "recommendations": farmer_tips
        },
        "public_advisory": {
            "outdoor_activity": recommendations[0] if recommendations else "Normal activities.",
            "recommendations": public_tips
        },
        "advisory_source": "WeatherGPT Advisory",
        "disclaimer": "WeatherGPT Advisory is automated meteorological decision support and does not replace official IMD emergency bulletins."
    }