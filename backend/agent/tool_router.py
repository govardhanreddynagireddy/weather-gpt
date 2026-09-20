# =====================================================
# WEATHERGPT+ TOOL ROUTER
# =====================================================


def route_tool(message):

    message=message.lower().strip()


    # =================================================
    # RAG / IMD QUESTIONS
    # =================================================

    rag_keywords=[
        "imd",
        "warning",
        "bulletin",
        "rainfall forecast",
        "rain forecast",
        "weather warning",
        "heavy rainfall",
        "forecast bulletin",
        "official forecast",
        "imd forecast",
        "imd warning",
        "weather alert",
        "cyclone alert",
        "storm warning",
        "హెచ్చరిక",
        "బులెటిన్",
        "వర్షపాతం",
        "తుఫాను",
        "అలర్ట్"
    ]


    # =================================================
    # GRU NEXT-HOUR / MODEL PREDICTION
    # =================================================

    gru_keywords=[
        "next hour",
        "next-hour",
        "next temperature",
        "temperature prediction",
        "predict temperature",
        "predict weather",
        "gru",
        "model prediction",
        "future temperature",
        "predicted temperature",
        "తరువాతి గంట",
        "తదుపరి గంట",
        "ఉష్ణోగ్రత అంచనా",
        "మోడల్ అంచనా"
    ]


    # =================================================
    # HISTORICAL WEATHER
    # =================================================

    historical_keywords=[
        "historical",
        "history",
        "last year",
        "previous year",
        "past weather",
        "historical weather",
        "past temperature",
        "historical temperature",
        "చరిత్ర",
        "గత సంవత్సరం",
        "గత వాతావరణం"
    ]


    # =================================================
    # RISK ANALYSIS
    # =================================================

    risk_keywords=[
        "risk",
        "danger",
        "impact",
        "crop risk",
        "agriculture risk",
        "flood risk",
        "heat risk",
        "rain risk",
        "weather risk",
        "farmer",
        "farmers",
        "ప్రమాదం",
        "రిస్క్",
        "ప్రభావం",
        "రైతు",
        "రైతులు",
        "వరద",
        "వేడి"
    ]


    has_rag=any(
        keyword in message
        for keyword in rag_keywords
    )

    has_gru=any(
        keyword in message
        for keyword in gru_keywords
    )

    has_historical=any(
        keyword in message
        for keyword in historical_keywords
    )

    has_risk=any(
        keyword in message
        for keyword in risk_keywords
    )


    # =================================================
    # COMBINED INTENT DETECTION
    # =================================================

    wants_current_and_gru=(
        has_gru and (
            "current weather" in message
            or "weather and" in message
            or "weather as well as" in message
            or "weather with" in message
            or "today and next hour" in message
            or "weather now" in message
            or "current conditions" in message
            or ("weather in" in message and has_gru)
            or ("current temperature" in message and has_gru)
            or ("వాతావరణం" in message and has_gru)
        )
    )

    wants_weather_and_rag=(
        has_rag and (
            "current weather" in message
            or "weather and" in message
            or "weather as well as" in message
            or "weather with" in message
            or "weather in" in message
            or "today's weather" in message
            or ("వాతావరణం" in message and has_rag)
        )
    )

    wants_gru_and_rag=(
        has_gru and has_rag
    )


    if wants_current_and_gru:
        return "weather_gru"

    if wants_weather_and_rag:
        return "weather_rag"

    if wants_gru_and_rag:
        return "gru_rag"


    # =================================================
    # SINGLE INTENT
    # =================================================

    if has_rag:
        return "rag"

    if has_gru:
        return "gru"

    if has_historical:
        return "historical"

    if has_risk:
        return "risk"


    # =================================================
    # GENERAL WEATHER
    # =================================================

    return "weather"


