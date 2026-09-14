from backend.agent.tool_router import route_tool

from backend.tools.gru_tool import (
    get_gru_info,
    predict_latest_temperature
)

from backend.services.ollama_service import (
    generate_response
)

from backend.tools.rag_tool import (
    search_weather
)

from backend.tools.weather_api_tool import (
    get_weather,
    get_weather_forecast
)


# =====================================================
# ORCHESTRATOR
# =====================================================

def orchestrate(message):

    # =================================================
    # VALIDATE MESSAGE
    # =================================================

    if not message or not message.strip():

        return {
            "success":False,
            "error":"Message cannot be empty."
        }

    message=message.strip()

    tool=route_tool(message)


    # =================================================
    # GENERAL WEATHER → OPENWEATHER → OLLAMA
    # =================================================

    if tool=="weather":

        # ---------------------------------------------
        # Get city
        # ---------------------------------------------

        city="Kurnool"

        if " in " in message.lower():

            city_part=message.lower().split(
                " in ",
                1
            )[1]

            city_part=city_part.replace(
                " tomorrow",
                ""
            ).replace(
                " today",
                ""
            ).replace(
                "?",
                ""
            ).strip()

            if city_part:

                city=city_part


        # ---------------------------------------------
        # Determine forecast day
        # ---------------------------------------------

        days_ahead=0

        if "tomorrow" in message.lower():

            days_ahead=1


        # ---------------------------------------------
        # Get weather data
        # ---------------------------------------------

        try:

            if days_ahead==1:

                weather_data=get_weather_forecast(
                    city,
                    days_ahead
                )

            else:

                weather_data=get_weather(
                    city
                )

        except Exception as e:

            return {
                "success":False,
                "tool":"weather",
                "message":message,
                "error":str(e)
            }


        # ---------------------------------------------
        # Send weather data to Ollama
        # ---------------------------------------------

        prompt=f"""
You are WeatherGPT+, a helpful weather assistant.

Answer the user's question using ONLY the weather
data provided below.

IMPORTANT RULES:

1. Do not invent weather information.

2. Do not say that you do not have access to live
   weather data because actual weather data is
   provided below.

3. Answer the user's question directly.

4. If multiple forecast times are provided, summarize
   the weather conditions throughout the day.

5. Mention temperature, weather condition, humidity,
   wind speed and rainfall probability when available.

USER QUESTION:
{message}

WEATHER DATA:
{weather_data}

Now provide a clear, concise answer.
"""

        answer=generate_response(
            prompt
        )

        return {
            "success":True,
            "tool":"weather",
            "message":message,
            "weather_data":weather_data,
            "answer":answer
        }


    # =================================================
    # GRU → NEXT TEMPERATURE
    # =================================================

    if tool=="gru":

        prediction=predict_latest_temperature()

        return {
            "success":True,
            "tool":"gru",
            "message":message,
            "prediction":prediction,
            "model":get_gru_info(),
            "source":"era5_merged.nc"
        }


    # =================================================
    # RAG → FAISS → OLLAMA
    # =================================================

    if tool=="rag":

        results=search_weather(
            message
        )


        # -------------------------------------------------
        # No relevant documents
        # -------------------------------------------------

        if not results:

            return {
                "success":True,
                "tool":"rag",
                "message":message,
                "answer":(
                    "No relevant information was found "
                    "in the IMD documents."
                ),
                "sources":[]
            }


        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context=""

        for i,result in enumerate(
            results,
            1
        ):

            context+=f"""
SOURCE {i}

DOCUMENT:
{result["source"]}

CHUNK:
{result["chunk_id"]}

RELEVANCE:
{result["score"]}

TEXT:
{result["text"]}

"""


        # -------------------------------------------------
        # RAG → OLLAMA
        # -------------------------------------------------

        prompt=f"""
You are WeatherGPT+, an official weather information
assistant.

Your task is to answer the user's question using ONLY
the IMD weather context provided below.

IMPORTANT RULES:

1. Carefully read ALL the IMD context before answering.

2. If the answer is present anywhere in the context,
   you MUST answer the user's question.

3. Do NOT say that the information is unavailable if
   the information is explicitly present in the context.

4. Do NOT use outside knowledge.

5. Do NOT invent weather information.

6. If multiple documents contain similar information,
   prefer the latest IMD document.

7. Pay attention to the actual forecast date mentioned
   in the document.

8. If the user asks about "tomorrow", determine the
   correct forecast date from the IMD context.

9. Mention the rainfall intensity and affected locations
   when they are available.

10. If warning information is present, explain it
    directly instead of refusing to answer.

11. Keep the answer concise and easy to understand.

USER QUESTION:
{message}

IMD WEATHER CONTEXT:
{context}

Now answer the user's question directly.
"""

        answer=generate_response(
            prompt
        )


        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources=[]

        for result in results:

            sources.append({

                "source":result["source"],

                "chunk_id":result["chunk_id"],

                "score":result["score"]

            })


        return {

            "success":True,

            "tool":"rag",

            "message":message,

            "answer":answer,

            "sources":sources

        }


    # =================================================
    # HISTORICAL
    # =================================================

    if tool=="historical":

        return {

            "success":True,

            "tool":"historical",

            "message":message,

            "answer":(
                "Historical weather analysis "
                "is not connected yet."
            )

        }


    # =================================================
    # RISK
    # =================================================

    if tool=="risk":

        return {

            "success":True,

            "tool":"risk",

            "message":message,

            "answer":(
                "Weather risk analysis "
                "is not connected yet."
            )

        }


    # =================================================
    # FALLBACK → OLLAMA
    # =================================================

    prompt=f"""
You are WeatherGPT+.

User question:
{message}

Give a helpful and concise answer.

Do not invent live weather information.
"""

    answer=generate_response(
        prompt
    )

    return {

        "success":True,

        "tool":"weather",

        "message":message,

        "answer":answer

    }