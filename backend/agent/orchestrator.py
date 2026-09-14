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
# EXTRACT CITY
# =====================================================

def extract_city(message):

    message_lower=message.lower().strip()

    message_lower=message_lower.replace(
        "?",
        ""
    ).strip()

    if " in " in message_lower:

        city=message_lower.split(
            " in ",
            1
        )[1].strip()

        words_to_remove=[
            " tomorrow",
            " today",
            " next hour",
            " right now",
            " now"
        ]

        for word in words_to_remove:

            city=city.replace(
                word,
                ""
            ).strip()

        if city and city!="undefined":

            return city.title()

    return "Kurnool"


# =====================================================
# ORCHESTRATOR
# =====================================================

def orchestrate(message):

    print()
    print("========================================")
    print("ORCHESTRATOR START")
    print("MESSAGE:",message)
    print("========================================")

    # =================================================
    # VALIDATE
    # =================================================

    if not message or not message.strip():

        return {
            "success":False,
            "error":"Message cannot be empty."
        }

    message=message.strip()

    # =================================================
    # ROUTING
    # =================================================

    print("DEBUG 0: ROUTING TOOL")

    tool=route_tool(message)

    print("DEBUG 0: SELECTED TOOL =",tool)


    # =================================================
    # WEATHER
    # =================================================

    if tool=="weather":

        print()
        print("========================================")
        print("WEATHER DEBUG")
        print("========================================")

        city=extract_city(message)

        print("DEBUG WEATHER 1: CITY =",city)

        days_ahead=0

        if "tomorrow" in message.lower():

            days_ahead=1

        print(
            "DEBUG WEATHER 2: DAYS AHEAD =",
            days_ahead
        )

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

            print(
                "DEBUG WEATHER ERROR:",
                str(e)
            )

            return {

                "success":False,

                "tool":"weather",

                "message":message,

                "error":str(e)

            }


        print("DEBUG WEATHER 3: WEATHER API DONE")

        # ---------------------------------------------
        # RAG
        # ---------------------------------------------

        rag_results=[]

        try:

            print("DEBUG WEATHER 4: STARTING RAG")

            rag_results=search_weather(
                message
            )

            print(
                "DEBUG WEATHER 5: RAG DONE"
            )

        except Exception as e:

            print(
                "DEBUG WEATHER RAG ERROR:",
                str(e)
            )

            rag_results=[]


        context=""

        for i,result in enumerate(
            rag_results,
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


        # ---------------------------------------------
        # OLLAMA
        # ---------------------------------------------

        prompt=f"""
You are WeatherGPT+, an AI weather assistant.

Answer the user's question using the information
provided below.

LIVE WEATHER DATA:
{weather_data}

IMD RAG CONTEXT:
{context}

USER QUESTION:
{message}

IMPORTANT RULES:

1. Use the live weather data for current conditions.

2. Use IMD RAG context when relevant.

3. Never invent weather values.

4. Never invent temperatures, rainfall or warnings.

5. If IMD context contains a relevant warning,
   mention it.

6. Keep the answer concise.

7. Clearly distinguish live weather from IMD
   forecast/warning information.

Now answer the user.
"""

        print("DEBUG WEATHER 6: STARTING OLLAMA")

        try:

            answer=generate_response(
                prompt
            )

            print("DEBUG WEATHER 7: OLLAMA DONE")

        except Exception as e:

            print(
                "DEBUG WEATHER OLLAMA ERROR:",
                str(e)
            )

            answer=(
                f"Current weather in {city}: "
                f"{weather_data}"
            )


        return {

            "success":True,

            "tool":"weather",

            "type":"city_weather",

            "message":message,

            "weather":weather_data,

            "weather_data":weather_data,

            "rag_sources":[

                {
                    "source":r["source"],
                    "chunk_id":r["chunk_id"],
                    "score":r["score"]
                }

                for r in rag_results

            ],

            "answer":answer

        }


    # =================================================
    # GRU → NEXT HOUR TEMPERATURE
    # NO OLLAMA
    # NO RAG
    # =================================================

    if tool=="gru":

        print()
        print("========================================")
        print("GRU DEBUG")
        print("========================================")

        print(
            "DEBUG GRU 1: GRU TOOL SELECTED"
        )

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        print(
            "DEBUG GRU 2: STARTING GRU PREDICTION"
        )

        try:

            prediction=(
                predict_latest_temperature()
            )

        except Exception as e:

            print(
                "DEBUG GRU ERROR:",
                str(e)
            )

            return {

                "success":False,

                "tool":"gru",

                "message":message,

                "error":str(e)

            }


        print(
            "DEBUG GRU 3: GRU PREDICTION DONE"
        )

        print(
            "DEBUG GRU PREDICTION:",
            prediction
        )


        # ---------------------------------------------
        # Model information
        # ---------------------------------------------

        print(
            "DEBUG GRU 4: GETTING MODEL INFO"
        )

        model_info=get_gru_info()

        print(
            "DEBUG GRU 5: MODEL INFO DONE"
        )


        # ---------------------------------------------
        # Extract temperature
        # ---------------------------------------------

        predicted_temperature=prediction

        if isinstance(
            prediction,
            dict
        ):

            predicted_temperature=prediction.get(
                "prediction_celsius",
                prediction.get(
                    "predicted_temperature_celsius",
                    prediction.get(
                        "prediction",
                        prediction
                    )
                )
            )


        print(
            "DEBUG GRU 6: PREPARING RESPONSE"
        )


        # ---------------------------------------------
        # DIRECT RESPONSE
        #
        # NO RAG
        # NO OLLAMA
        # ---------------------------------------------

        answer=(
            "The WeatherGRU model predicts the "
            f"next-hour temperature to be "
            f"{predicted_temperature:.2f} °C."
        )


        print(
            "DEBUG GRU 7: RESPONSE READY"
        )

        print(
            "DEBUG GRU 8: RETURNING RESPONSE"
        )


        return {

            "success":True,

            "tool":"gru",

            "type":"gru_prediction",

            "message":message,

            "prediction":prediction,

            "predicted_temperature_celsius":
                predicted_temperature,

            "model":model_info,

            "source":"era5_merged.nc",

            "prediction_horizon":
                "next_hour",

            "answer":answer

        }


    # =================================================
    # RAG → FAISS → OLLAMA
    # =================================================

    if tool=="rag":

        print()
        print("========================================")
        print("RAG DEBUG")
        print("========================================")

        print(
            "DEBUG RAG 1: SEARCHING DOCUMENTS"
        )

        try:

            results=search_weather(
                message
            )

        except Exception as e:

            return {

                "success":False,

                "tool":"rag",

                "message":message,

                "error":str(e)

            }


        print(
            "DEBUG RAG 2: SEARCH DONE"
        )


        if not results:

            return {

                "success":True,

                "tool":"rag",

                "type":"rag_response",

                "message":message,

                "answer":(
                    "No relevant information was "
                    "found in the IMD documents."
                ),

                "sources":[]

            }


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


        prompt=f"""
You are WeatherGPT+, an official weather information
assistant.

Use ONLY the IMD context below.

USER QUESTION:
{message}

IMD WEATHER CONTEXT:
{context}

RULES:

1. Read all retrieved context.

2. Use the latest relevant document.

3. Pay attention to forecast dates.

4. Do not invent weather information.

5. Mention warnings when available.

6. Keep the answer concise.

Now answer the user directly.
"""


        print(
            "DEBUG RAG 3: STARTING OLLAMA"
        )

        try:

            answer=generate_response(
                prompt
            )

            print(
                "DEBUG RAG 4: OLLAMA DONE"
            )

        except Exception as e:

            print(
                "DEBUG RAG OLLAMA ERROR:",
                str(e)
            )

            answer=(
                "Relevant information was found "
                "in the IMD documents, but the AI "
                "response service is unavailable."
            )


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

            "type":"rag_response",

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
    # FALLBACK
    # =================================================

    print(
        "DEBUG FALLBACK: STARTING OLLAMA"
    )

    try:

        answer=generate_response(
            f"""
You are WeatherGPT+.

User question:
{message}

Give a helpful and concise answer.

Do not invent live weather information.
"""
        )

    except Exception:

        answer=(
            "I couldn't generate an AI response "
            "because the response service is unavailable."
        )


    return {

        "success":True,

        "tool":"fallback",

        "message":message,

        "answer":answer

    }