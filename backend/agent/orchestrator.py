from backend.agent.tool_router import route_tool

from backend.tools.gru_tool import (
    get_gru_info,
    predict_latest_temperature
)

from backend.tools.rag_tool import (
    search_weather
)

# =====================================================
# FIXED: this used to import from backend.tools.weather_api_tool,
# which returns {location, temperature, humidity, wind_speed,
# weather, rainfall}. Everything below in this file (and the
# frontend) reads temperature_celsius / humidity_percent /
# wind_speed_kmh / weather_description / precipitation_mm,
# which is the schema returned by weather_tool, not
# weather_api_tool. That mismatch is what produced "--" for
# every field. weather_tool is also what main.py's /weather/{city}
# endpoint already uses, so this makes the whole app consistent.
# =====================================================

from backend.tools.weather_tool import (
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
# BUILD TOMORROW FORECAST SUMMARY
#
# weather_tool.get_weather_forecast() returns a list of
# 3-hourly OpenWeatherMap entries under "forecasts". The
# frontend needs a single aggregated object instead:
# date, max_temperature_celsius, min_temperature_celsius,
# weather_description, precipitation_mm.
# =====================================================

def summarize_forecast(city,forecast_data):

    entries=forecast_data.get(
        "forecasts",
        []
    )

    if not entries:

        return {

            "city":city,

            "date":forecast_data.get(
                "date"
            ),

            "max_temperature_celsius":"--",

            "min_temperature_celsius":"--",

            "weather_description":"--",

            "precipitation_mm":0

        }


    temperatures=[
        entry["temperature"]
        for entry in entries
        if "temperature" in entry
    ]

    max_temperature=round(
        max(temperatures),
        2
    ) if temperatures else "--"

    min_temperature=round(
        min(temperatures),
        2
    ) if temperatures else "--"

    total_precipitation=round(
        sum(
            entry.get("rainfall",0)
            for entry in entries
        ),
        2
    )


    # -----------------------------------------------
    # Representative condition: the entry closest to
    # local midday, since that best represents "the"
    # weather for the day rather than an early-morning
    # or late-night reading.
    # -----------------------------------------------

    def hour_distance_from_noon(entry):

        try:

            hour=int(
                entry["time"].split(":")[0]
            )

        except (KeyError,ValueError,IndexError):

            return 99

        return abs(hour-12)


    representative=min(
        entries,
        key=hour_distance_from_noon
    )

    condition=representative.get(
        "weather",
        "--"
    )


    return {

        "city":city,

        "date":forecast_data.get(
            "date"
        ),

        "max_temperature_celsius":max_temperature,

        "min_temperature_celsius":min_temperature,

        "weather_description":condition,

        "precipitation_mm":total_precipitation

    }


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

    print(
        "DEBUG 0: SELECTED TOOL =",
        tool
    )


    # =================================================
    # WEATHER
    # =================================================

    if tool=="weather":

        print()
        print("========================================")
        print("WEATHER DEBUG")
        print("========================================")

        city=extract_city(message)

        print(
            "DEBUG WEATHER 1: CITY =",
            city
        )


        days_ahead=0

        if "tomorrow" in message.lower():

            days_ahead=1


        print(
            "DEBUG WEATHER 2: DAYS AHEAD =",
            days_ahead
        )


        # =================================================
        # FETCH WEATHER
        # =================================================

        try:

            if days_ahead==1:

                print(
                    "DEBUG WEATHER 3: GETTING FORECAST"
                )

                weather_data=get_weather_forecast(
                    city,
                    days_ahead
                )

            else:

                print(
                    "DEBUG WEATHER 3: GETTING CURRENT WEATHER"
                )

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


        print(
            "DEBUG WEATHER 4: WEATHER API DONE"
        )

        print(
            "DEBUG WEATHER DATA:",
            weather_data
        )


        # =================================================
        # CURRENT WEATHER
        # =================================================

        if days_ahead==0:

            temperature=weather_data.get(
                "temperature_celsius",
                "--"
            )

            humidity=weather_data.get(
                "humidity_percent",
                "--"
            )

            wind=weather_data.get(
                "wind_speed_kmh",
                "--"
            )

            condition=weather_data.get(
                "weather_description",
                "--"
            )

            rainfall=weather_data.get(
                "precipitation_mm",
                "--"
            )


            answer=(
                f"Current weather in {city}:\n\n"
                f"🌡️ Temperature: {temperature} °C\n"
                f"💧 Humidity: {humidity}%\n"
                f"💨 Wind: {wind} km/h\n"
                f"🌧️ Rain: {rainfall} mm\n"
                f"🌤️ Condition: {condition}"
            )


            print(
                "DEBUG WEATHER 5: RESPONSE READY"
            )


            return {

                "success":True,

                "tool":"weather",

                "type":"city_weather",

                "message":message,

                "weather":weather_data,

                "weather_data":weather_data,

                "rag_sources":[],

                "answer":answer

            }


        # =================================================
        # TOMORROW FORECAST
        # =================================================

        print(
            "DEBUG WEATHER 5: BUILDING FORECAST SUMMARY"
        )

        forecast_summary=summarize_forecast(
            city,
            weather_data
        )

        print(
            "DEBUG WEATHER FORECAST SUMMARY:",
            forecast_summary
        )


        answer=(
            f"Tomorrow's forecast for {city} "
            f"({forecast_summary['date']}):\n\n"
            f"🌡️ Max: {forecast_summary['max_temperature_celsius']} °C\n"
            f"🌡️ Min: {forecast_summary['min_temperature_celsius']} °C\n"
            f"🌧️ Precipitation: {forecast_summary['precipitation_mm']} mm\n"
            f"🌤️ Condition: {forecast_summary['weather_description']}"
        )


        return {

            "success":True,

            "tool":"weather",

            "type":"tomorrow_forecast",

            "message":message,

            "forecast":forecast_summary,

            "raw_forecast":weather_data,

            "answer":answer

        }


    # =================================================
    # GRU → NEXT HOUR
    # NO OLLAMA
    # =================================================

    if tool=="gru":

        print()
        print("========================================")
        print("GRU DEBUG")
        print("========================================")


        print(
            "DEBUG GRU 1: GRU TOOL SELECTED"
        )


        # =================================================
        # PREDICTION
        # =================================================

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


        # =================================================
        # MODEL INFORMATION
        # =================================================

        print(
            "DEBUG GRU 4: GETTING MODEL INFO"
        )

        model_info=get_gru_info()


        print(
            "DEBUG GRU 5: MODEL INFO DONE"
        )


        # =================================================
        # EXTRACT TEMPERATURE
        # =================================================

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
            "DEBUG GRU 6: PREDICTED TEMPERATURE =",
            predicted_temperature
        )


        # =================================================
        # DIRECT RESPONSE
        # =================================================

        answer=(
            "📈 Next-hour temperature prediction:\n\n"
            f"The WeatherGRU model predicts the "
            f"temperature to be approximately "
            f"{predicted_temperature:.2f} °C."
        )


        print(
            "DEBUG GRU 7: RESPONSE READY"
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
    # RAG → FAISS
    # NO OLLAMA
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

            print(
                "DEBUG RAG ERROR:",
                str(e)
            )

            return {

                "success":False,

                "tool":"rag",

                "type":"rag_error",

                "message":message,

                "error":str(e)

            }


        print(
            "DEBUG RAG 2: SEARCH DONE"
        )

        print(
            "DEBUG RAG RESULTS:",
            len(results)
        )


        # =================================================
        # NO RESULTS
        # =================================================

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


        # =================================================
        # DIRECT RAG RESPONSE
        # =================================================

        answer="Relevant IMD information:\n\n"


        for i,result in enumerate(
            results,
            1
        ):

            answer+=(
                f"Source {i}:\n"
                f"{result['text']}\n\n"
            )


        sources=[]


        for result in results:

            sources.append({

                "source":
                    result["source"],

                "chunk_id":
                    result["chunk_id"],

                "score":
                    result["score"]

            })


        print(
            "DEBUG RAG 3: RESPONSE READY"
        )


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
    # NO OLLAMA
    # =================================================

    print(
        "DEBUG FALLBACK: NO TOOL MATCHED"
    )


    return {

        "success":True,

        "tool":"fallback",

        "type":"fallback_response",

        "message":message,

        "answer":(
            "I can help with:\n\n"
            "🌤️ Current weather\n"
            "🔮 Tomorrow's forecast\n"
            "📈 Next-hour temperature prediction\n"
            "📄 IMD weather information"
        )

    }
