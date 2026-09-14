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
        "storm warning"

    ]


    for keyword in rag_keywords:

        if keyword in message:

            return "rag"


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
        "future temperature"

    ]


    for keyword in gru_keywords:

        if keyword in message:

            return "gru"


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
        "historical temperature"

    ]


    for keyword in historical_keywords:

        if keyword in message:

            return "historical"


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
        "weather risk"

    ]


    for keyword in risk_keywords:

        if keyword in message:

            return "risk"


    # =================================================
    # GENERAL WEATHER
    # =================================================

    return "weather"

