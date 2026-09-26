"""
Activity & Purpose Decision Analysis Engine for WeatherGPT+
Evaluates meteorological suitability for activities (farming, travel, outdoor events, etc.)
against calibrated thresholds, producing grounded assessments and multi-source attributions.
"""

from typing import Dict, Any, List, Optional
from backend.agent.activity_registry import ACTIVITY_REGISTRY, find_matching_activity


def evaluate_activity_suitability(
    activity: Dict[str, Any],
    weather_data: Optional[Dict[str, Any]] = None,
    forecast_summary: Optional[Dict[str, Any]] = None,
    risk_data: Optional[Dict[str, Any]] = None,
    imd_warnings: Optional[List[Dict[str, Any]]] = None,
    time_intent: str = "current",
    city: str = "Kurnool",
    language: str = "en"
) -> Dict[str, Any]:
    """
    Evaluates weather suitability for an activity based on real meteorological parameters,
    calibrated safety/operational thresholds, and official IMD warnings.
    """
    act_key = activity.get("key", "general")
    act_name = activity.get("name", "Activity")
    thresholds = activity.get("thresholds", {})

    # Extract numerical parameters
    is_forecast = (time_intent == "tomorrow") or (forecast_summary is not None and time_intent != "current")

    if is_forecast and forecast_summary:
        temp_val = forecast_summary.get("max_temperature_celsius")
        rain_val = forecast_summary.get("precipitation_mm", 0.0)
        condition_val = forecast_summary.get("weather_description", "Partly Cloudy")
        horizon_desc = f"Tomorrow ({forecast_summary.get('date', 'forecast')})"
        # If forecast has no wind/humidity, safely fall back to current observations
        wind_val = (weather_data or {}).get("wind_speed_kmh", 8.0)
        humidity_val = (weather_data or {}).get("humidity_percent", 55.0)
    else:
        temp_val = (weather_data or {}).get("temperature_celsius", 28.0)
        rain_val = (weather_data or {}).get("precipitation_mm", 0.0)
        wind_val = (weather_data or {}).get("wind_speed_kmh", 10.0)
        humidity_val = (weather_data or {}).get("humidity_percent", 60.0)
        condition_val = (weather_data or {}).get("weather_description", "Fair")
        horizon_desc = "Current observation"

    try:
        temp = float(temp_val) if temp_val not in ["--", None] else 28.0
    except (ValueError, TypeError):
        temp = 28.0

    try:
        rain = float(rain_val) if rain_val not in ["--", None] else 0.0
    except (ValueError, TypeError):
        rain = 0.0

    try:
        wind = float(wind_val) if wind_val not in ["--", None] else 10.0
    except (ValueError, TypeError):
        wind = 10.0

    try:
        humidity = float(humidity_val) if humidity_val not in ["--", None] else 60.0
    except (ValueError, TypeError):
        humidity = 60.0

    condition_lower = str(condition_val).lower()

    # Default evaluation state
    suitability = "FAVORABLE"
    factors: List[str] = []
    recs: List[str] = []

    # -------------------------------------------------------------------------
    # DOMAIN / ACTIVITY SPECIFIC EVALUATIONS
    # -------------------------------------------------------------------------

    if act_key == "pesticide_spraying":
        max_wind = thresholds.get("max_wind_kmh", 15.0)
        max_rain = thresholds.get("max_rain_mm", 0.2)
        max_temp = thresholds.get("max_temp_c", 36.0)

        if rain > max_rain:
            suitability = "UNFAVORABLE"
            factors.append(f"Precipitation ({rain:.1f} mm > {max_rain:.1f} mm threshold) will cause chemical wash-off and poor efficacy.")
            recs.append("Postpone spraying until weather clears and crop foliage dries completely.")
        elif wind > max_wind:
            suitability = "UNFAVORABLE"
            factors.append(f"Wind speed ({wind:.1f} km/h > {max_wind:.1f} km/h limit) poses high spray drift risk to non-target areas.")
            recs.append(f"Wait for calm periods when wind speed subsides below {max_wind:.0f} km/h (early morning or late evening).")
        elif temp > max_temp:
            suitability = "MARGINAL"
            factors.append(f"High temperature ({temp:.1f} °C > {max_temp:.0f} °C) accelerates droplet evaporation and potential foliar scorching.")
            recs.append("Spray exclusively during cooler early morning hours (6 AM - 9 AM).")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Wind speed ({wind:.1f} km/h <= {max_wind:.0f} km/h limit) and no significant rain ({rain:.1f} mm) provide safe conditions.")
            recs.append("Ideal conditions for foliar and pesticide spraying. Maintain standard safety gear.")

    elif act_key == "crop_drying":
        max_rain = thresholds.get("max_rain_mm", 0.0)
        max_hum = thresholds.get("max_humidity_percent", 72.0)

        if rain > max_rain or any(w in condition_lower for w in ["rain", "drizzle", "shower", "thunderstorm"]):
            suitability = "UNFAVORABLE"
            factors.append(f"Precipitation ({rain:.1f} mm) or wet conditions will dampen grains and trigger mold/spoilage.")
            recs.append("Do not leave harvested produce in the open. Store in elevated, covered shelters or use tarpaulins.")
        elif humidity > max_hum:
            suitability = "MARGINAL"
            factors.append(f"High ambient humidity ({humidity:.0f}% > {max_hum:.0f}%) will substantially prolong drying times.")
            recs.append("Ensure thin spreading and frequent turning of produce to facilitate moisture loss.")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Zero rainfall ({rain:.1f} mm), low humidity ({humidity:.0f}%), and sunny conditions facilitate rapid drying.")
            recs.append("Optimal conditions for open-air sun drying of grains, chillies, and pulses.")

    elif act_key == "umbrella_rain_gear":
        has_rain_indicator = rain >= 0.2 or any(w in condition_lower for w in ["rain", "shower", "drizzle", "wet", "thunderstorm"])
        if has_rain_indicator:
            suitability = "RECOMMENDED"
            factors.append(f"Precipitation ({rain:.1f} mm) or rainy condition ('{condition_val}') is expected.")
            recs.append("Carry an umbrella, raincoat, or waterproof backpack cover.")
        else:
            suitability = "NOT_NEEDED"
            factors.append(f"Dry skies with negligible precipitation ({rain:.1f} mm) and condition '{condition_val}'.")
            recs.append("No umbrella or rain gear is currently required.")

    elif act_key == "crop_harvesting":
        max_rain = thresholds.get("max_rain_mm", 1.0)
        max_wind = thresholds.get("max_wind_kmh", 28.0)

        if rain > max_rain:
            suitability = "UNFAVORABLE"
            factors.append(f"Rainfall ({rain:.1f} mm) causes wet harvest, machinery slippage, and high post-harvest decay risk.")
            recs.append("Delay combining or manual harvesting until fields dry out.")
        elif wind > max_wind:
            suitability = "MARGINAL"
            factors.append(f"Gusty winds ({wind:.1f} km/h) increase lodging hazard during mechanical reaping.")
            recs.append("Exercise care during harvesting operations.")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Dry conditions ({rain:.1f} mm rain) allow smooth mechanical and manual harvesting.")
            recs.append("Favorable window to harvest and transport crops safely.")

    elif act_key == "irrigation":
        rain_saving = thresholds.get("rain_saving_mm", 5.0)
        if rain >= rain_saving:
            suitability = "UNFAVORABLE" # Don't irrigate
            factors.append(f"Significant natural rainfall ({rain:.1f} mm >= {rain_saving:.1f} mm) satisfies crop moisture requirements.")
            recs.append("Suspend artificial irrigation to conserve water and prevent waterlogging.")
        else:
            suitability = "FAVORABLE" # Irrigate
            factors.append(f"Natural rainfall ({rain:.1f} mm) is negligible; soil water replenishment is needed.")
            recs.append("Proceed with regular irrigation schedule according to crop stage.")

    elif act_key in ["travel_driving", "daily_commute"]:
        heavy_rain = thresholds.get("heavy_rain_mm", 20.0)
        high_wind = thresholds.get("high_wind_kmh", 35.0)

        if rain >= heavy_rain:
            suitability = "UNFAVORABLE"
            factors.append(f"Heavy rainfall ({rain:.1f} mm) can cause road waterlogging, hydroplaning, and poor visibility.")
            recs.append("Avoid non-essential highway travel; drive with low beams and reduced speed.")
        elif wind >= high_wind:
            suitability = "MARGINAL"
            factors.append(f"Strong crosswinds ({wind:.1f} km/h) affect high-profile vehicles and two-wheelers.")
            recs.append("Exercise caution on elevated highways and bridges.")
        elif rain > 0.5:
            suitability = "MARGINAL"
            factors.append(f"Light to moderate rain ({rain:.1f} mm) creates slick road surfaces.")
            recs.append("Maintain increased following distance.")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Clear weather with dry roads ({rain:.1f} mm rain, {wind:.1f} km/h wind).")
            recs.append("Normal driving conditions; follow standard traffic safety rules.")

    elif act_key == "outdoor_event":
        if rain > 1.0:
            suitability = "UNFAVORABLE"
            factors.append(f"Rainfall ({rain:.1f} mm) will disrupt open-air seating and sound equipment.")
            recs.append("Arrange waterproof shamianas, covered canopies, or an indoor alternative.")
        elif temp >= 38.0:
            suitability = "MARGINAL"
            factors.append(f"Elevated temperatures ({temp:.1f} °C) may cause discomfort for guests.")
            recs.append("Provide adequate shade, water dispensers, and mist fans.")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Pleasant conditions with {temp:.1f} °C and zero rain disruption.")
            recs.append("Excellent weather for open-air gatherings and photography.")

    elif act_key == "heat_exposure":
        if temp >= 40.0:
            suitability = "UNFAVORABLE"
            factors.append(f"Extreme heat ({temp:.1f} °C) carries severe risk of heat exhaustion and heatstroke.")
            recs.append("Avoid direct outdoor exposure from 11 AM to 4 PM; consume oral rehydration fluids.")
        elif temp >= 35.0:
            suitability = "MARGINAL"
            factors.append(f"Warm weather ({temp:.1f} °C) increases dehydration rates.")
            recs.append("Stay hydrated and wear loose cotton clothing.")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Temperatures ({temp:.1f} °C) remain within comfortable limits.")
            recs.append("Safe for regular daytime outdoor activities.")

    else:
        # Generic activity fallback
        if rain > 1.0 or wind > 25.0:
            suitability = "MARGINAL"
            factors.append(f"Active weather parameters: Rain {rain:.1f} mm, Wind {wind:.1f} km/h.")
            recs.append("Take normal outdoor precautions.")
        else:
            suitability = "FAVORABLE"
            factors.append(f"Mild conditions: {temp:.1f} °C, Rain {rain:.1f} mm, Wind {wind:.1f} km/h.")
            recs.append("Conditions appear favorable for outdoor activities.")

    # -------------------------------------------------------------------------
    # OFFICIAL WARNINGS INTEGRATION
    # -------------------------------------------------------------------------
    official_bulletin_str = "No active severe weather warnings reported by IMD for this district."
    if imd_warnings and len(imd_warnings) > 0:
        first_bulletin = imd_warnings[0]
        text_snippet = first_bulletin.get("text", "").strip()[:180]
        if text_snippet:
            official_bulletin_str = f"IMD Bulletin: {text_snippet}..."

    # -------------------------------------------------------------------------
    # MULTI-SOURCE ATTRIBUTIONS
    # -------------------------------------------------------------------------
    attributions = {
        "observation": f"Measured ({city}): {temp:.1f} °C, Wind {wind:.1f} km/h, Rain {rain:.1f} mm, Humidity {humidity:.0f}%, Condition '{condition_val}'",
        "forecast": f"{horizon_desc}: Max {temp:.1f} °C, Rain {rain:.1f} mm, Sky '{condition_val}'",
        "advisory": f"WeatherGPT+ Decision Support: Evaluated for '{act_name}' — Status: {suitability}",
        "official_warnings": official_bulletin_str
    }

    # -------------------------------------------------------------------------
    # DETERMINISTIC NATURAL LANGUAGE GENERATION (ENGLISH & TELUGU)
    # -------------------------------------------------------------------------
    status_emoji = "✅" if suitability in ["FAVORABLE", "NOT_NEEDED"] else ("⚠️" if suitability == "MARGINAL" else "❌")

    # Direct answer phrase
    if act_key == "pesticide_spraying":
        direct_en = f"Yes, conditions are favorable for spraying in {city}." if suitability == "FAVORABLE" else f"No, pesticide spraying is not advised in {city}."
        direct_te = f"అవును, {city} లో మందు కొట్టడానికి వాతావరణం అనుకూలంగా ఉంది." if suitability == "FAVORABLE" else f"లేదు, ప్రస్తుతం {city} లో మందు కొట్టడం మంచిది కాదు."
    elif act_key == "crop_drying":
        direct_en = f"Yes, you can dry your crops in {city}." if suitability == "FAVORABLE" else f"No, open-air crop drying is not recommended in {city}."
        direct_te = f"అవును, {city} లో పంట ఆరబెట్టడానికి వాతావరణం బాగుంది." if suitability == "FAVORABLE" else f"లేదు, {city} లో పంట ఆరబెట్టడం సురక్షితం కాదు."
    elif act_key == "umbrella_rain_gear":
        direct_en = f"Yes, you should carry an umbrella in {city}." if suitability == "RECOMMENDED" else f"No, you do not need an umbrella in {city}."
        direct_te = f"అవును, {city} లో గొడుగు లేదా రెయిన్‌కోట్ తీసుకెళ్లడం మంచిది." if suitability == "RECOMMENDED" else f"లేదు, {city} లో ప్రస్తుతం గొడుగు అవసరం లేదు."
    elif act_key == "travel_driving":
        direct_en = f"Travel conditions in/to {city} are clear and favorable." if suitability == "FAVORABLE" else f"Exercise caution while driving or traveling to {city}."
        direct_te = f"{city} ప్రయాణానికి వాతావరణం అనుకూలంగా ఉంది." if suitability == "FAVORABLE" else f"{city} ప్రయాణంలో జాగ్రత్త వహించండి."
    else:
        direct_en = f"Conditions for {act_name.lower()} in {city} are evaluated as {suitability.lower()}."
        direct_te = f"{city} లో {act_name} కోసం వాతావరణం {suitability} గా అంచనా వేయబడింది."

    factors_str_en = "\n".join(f"• {f}" for f in factors)
    recs_str_en = "\n".join(f"• {r}" for r in recs)

    deterministic_answer_en = f"""{status_emoji} **{direct_en}**

📊 **Meteorological Assessment ({horizon_desc}):**
{factors_str_en}

💡 **Actionable Guidance:**
{recs_str_en}

---
🔍 **Source Attributions:**
• **Observations:** Temp {temp:.1f} °C, Rain {rain:.1f} mm, Wind {wind:.1f} km/h
• **WeatherGPT+ Advisory:** Threshold-based operational decision support
• **Official Warnings:** {official_bulletin_str}"""

    # Telugu deterministic formatting
    te_factors = factors_str_en.replace("Precipitation", "వర్షపాతం").replace("Wind speed", "గాలి వేగం").replace("threshold", "పరిమితి")
    te_recs = recs_str_en

    deterministic_answer_te = f"""{status_emoji} **{direct_te}**

📊 **వాతావరణ విశ్లేషణ ({horizon_desc}):**
{te_factors}

💡 **సూచనలు:**
{te_recs}

---
🔍 **సమాచార ఆధారాలు:**
• **పరిశీలనలు:** ఉష్ణోగ్రత {temp:.1f} °C, వర్షపాతం {rain:.1f} mm, గాలి {wind:.1f} km/h
• **WeatherGPT+ సలహా:** శాస్త్రీయ వాతావరణ ప్రమాణాల ఆధారంగా రూపొందించిన నిర్ణయం
• **IMD అధికారిక హెచ్చరికలు:** {official_bulletin_str}"""

    selected_answer = deterministic_answer_te if language == "te" else deterministic_answer_en

    return {
        "activity_key": act_key,
        "activity_name": act_name,
        "domain": activity.get("domain", "general"),
        "decision_context": activity.get("decision_context", "general"),
        "suitability": suitability,
        "time_intent": time_intent,
        "city": city,
        "parameters": {
            "temperature_celsius": temp,
            "precipitation_mm": rain,
            "wind_speed_kmh": wind,
            "humidity_percent": humidity,
            "condition": condition_val
        },
        "key_factors": factors,
        "recommendations": recs,
        "attributions": attributions,
        "direct_answer": direct_en if language == "en" else direct_te,
        "answer": selected_answer
    }
