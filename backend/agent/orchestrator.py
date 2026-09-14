import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Guard against native access violation in PyTorch CUDA stream capture on CPU/Windows
import torch
if hasattr(torch, "cuda"):
    torch.cuda.is_current_stream_capturing = lambda: False
    if hasattr(torch.cuda, "graphs"):
        torch.cuda.graphs.is_current_stream_capturing = lambda: False

try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_cuda_stream_capturing = lambda: False
except Exception:
    pass

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

    if not message:
        return "Kurnool"

    text = message.strip()
    for char in "?!.,:;\"'()[]{}":
        text = text.replace(char, " ")

    text_lower = " " + " ".join(text.lower().split()) + " "
    city = None

    for prep in [" in ", " for ", " of ", " at "]:
        if prep in text_lower:
            parts = text_lower.split(prep, 1)
            candidate = parts[1].strip()
            for stop in [" tomorrow", " today", " next hour", " right now", " now", " and ", " please"]:
                if stop in candidate:
                    candidate = candidate.split(stop, 1)[0].strip()
            if candidate and candidate.lower() != "undefined":
                city = candidate
                break

    if not city:
        for suffix in [" weather", " forecast", " temperature"]:
            if suffix in text_lower:
                candidate = text_lower.split(suffix, 1)[0].strip()
                words = candidate.split()
                if words:
                    last_word = words[-1]
                    if last_word not in ["the", "a", "an", "what", "is", "show", "get", "tell", "current", "tomorrow"]:
                        city = last_word

    if city and city.lower() != "undefined" and len(city) > 1:
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
            "error":"Message cannot be empty.",
            "answer":"⚠️ Message cannot be empty."
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
    # COMBINED: WEATHER + GRU
    # =================================================

    if tool=="weather_gru":

        city=extract_city(message)
        weather_data=None
        weather_err=None
        prediction=None
        predicted_temp=None
        gru_err=None

        try:
            weather_data=get_weather(city)
        except Exception as e:
            weather_err=str(e)

        try:
            prediction=predict_latest_temperature()
            if isinstance(prediction,dict):
                predicted_temp=prediction.get(
                    "prediction_celsius",
                    prediction.get(
                        "predicted_temperature_celsius",
                        prediction.get("prediction",0.0)
                    )
                )
            else:
                predicted_temp=float(prediction)
        except Exception as e:
            gru_err=str(e)

        answer_parts=[]
        if weather_data:
            answer_parts.append(
                f"Current weather in {city}:\n\n"
                f"🌡️ Temperature: {weather_data.get('temperature_celsius', '--')} °C\n"
                f"💧 Humidity: {weather_data.get('humidity_percent', '--')}%\n"
                f"💨 Wind: {weather_data.get('wind_speed_kmh', '--')} km/h\n"
                f"🌧️ Rain: {weather_data.get('precipitation_mm', 0)} mm\n"
                f"🌤️ Condition: {weather_data.get('weather_description', '--')}"
            )
        elif weather_err:
            answer_parts.append(f"⚠️ Current weather unavailable for {city}: {weather_err}")

        if predicted_temp is not None:
            answer_parts.append(
                f"📈 Next-hour temperature prediction (WeatherGRU):\n\n"
                f"The model predicts the next-hour temperature to be approximately {predicted_temp:.2f} °C."
            )
        elif gru_err:
            answer_parts.append(f"⚠️ GRU prediction unavailable: {gru_err}")

        answer="\n\n".join(answer_parts)

        return {
            "success":weather_data is not None or predicted_temp is not None,
            "tool":"weather_gru",
            "type":"city_weather",
            "message":message,
            "weather":weather_data,
            "weather_data":weather_data,
            "prediction":prediction,
            "predicted_temperature_celsius":predicted_temp,
            "rag_sources":[],
            "answer":answer
        }


    # =================================================
    # COMBINED: WEATHER + RAG
    # =================================================

    if tool=="weather_rag":

        city=extract_city(message)
        days_ahead=1 if "tomorrow" in message.lower() else 0
        weather_data=None
        weather_err=None
        results=[]
        sources=[]
        rag_err=None

        try:
            if days_ahead==1:
                weather_data=get_weather_forecast(city,days_ahead=1)
            else:
                weather_data=get_weather(city)
        except Exception as e:
            weather_err=str(e)

        try:
            results=search_weather(message)
            if results:
                for r in results:
                    sources.append({
                        "source":r.get("source","IMD Document"),
                        "chunk_id":r.get("chunk_id",""),
                        "score":round(float(r.get("score",0.0)),4)
                    })
        except Exception as e:
            rag_err=str(e)

        answer_parts=[]
        if weather_data:
            if days_ahead==1:
                summary=summarize_forecast(city,weather_data)
                answer_parts.append(
                    f"Tomorrow's forecast for {city} ({summary['date']}):\n\n"
                    f"🌡️ Max: {summary['max_temperature_celsius']} °C\n"
                    f"🌡️ Min: {summary['min_temperature_celsius']} °C\n"
                    f"🌧️ Precipitation: {summary['precipitation_mm']} mm\n"
                    f"🌤️ Condition: {summary['weather_description']}"
                )
            else:
                answer_parts.append(
                    f"Current weather in {city}:\n\n"
                    f"🌡️ Temperature: {weather_data.get('temperature_celsius', '--')} °C\n"
                    f"💧 Humidity: {weather_data.get('humidity_percent', '--')}%\n"
                    f"💨 Wind: {weather_data.get('wind_speed_kmh', '--')} km/h\n"
                    f"🌧️ Rain: {weather_data.get('precipitation_mm', 0)} mm\n"
                    f"🌤️ Condition: {weather_data.get('weather_description', '--')}"
                )
        elif weather_err:
            answer_parts.append(f"⚠️ Weather data unavailable for {city}: {weather_err}")

        if results:
            rag_lines=["Relevant IMD official information:\n"]
            for i,r in enumerate(results[:2],1):
                rag_lines.append(f"Source {i} ({r.get('source','IMD')}):\n{r.get('text','').strip()}\n")
            answer_parts.append("\n".join(rag_lines))
        elif rag_err:
            answer_parts.append(f"⚠️ IMD document search unavailable: {rag_err}")

        answer="\n\n".join(answer_parts)

        return {
            "success":weather_data is not None or len(results)>0,
            "tool":"weather_rag",
            "type":"city_weather" if days_ahead==0 else "tomorrow_forecast",
            "message":message,
            "weather":weather_data if days_ahead==0 else None,
            "forecast":summarize_forecast(city,weather_data) if (days_ahead==1 and weather_data) else None,
            "sources":sources,
            "rag_sources":sources,
            "answer":answer
        }


    # =================================================
    # COMBINED: GRU + RAG
    # =================================================

    if tool=="gru_rag":

        prediction=None
        predicted_temp=None
        gru_err=None
        results=[]
        sources=[]
        rag_err=None

        try:
            prediction=predict_latest_temperature()
            if isinstance(prediction,dict):
                predicted_temp=prediction.get(
                    "prediction_celsius",
                    prediction.get(
                        "predicted_temperature_celsius",
                        prediction.get("prediction",0.0)
                    )
                )
            else:
                predicted_temp=float(prediction)
        except Exception as e:
            gru_err=str(e)

        try:
            results=search_weather(message)
            if results:
                for r in results:
                    sources.append({
                        "source":r.get("source","IMD Document"),
                        "chunk_id":r.get("chunk_id",""),
                        "score":round(float(r.get("score",0.0)),4)
                    })
        except Exception as e:
            rag_err=str(e)

        answer_parts=[]
        if predicted_temp is not None:
            answer_parts.append(
                f"📈 Next-hour temperature prediction (WeatherGRU):\n\n"
                f"The model predicts the temperature to be approximately {predicted_temp:.2f} °C."
            )
        elif gru_err:
            answer_parts.append(f"⚠️ GRU prediction unavailable: {gru_err}")

        if results:
            rag_lines=["Relevant IMD official information:\n"]
            for i,r in enumerate(results[:2],1):
                rag_lines.append(f"Source {i} ({r.get('source','IMD')}):\n{r.get('text','').strip()}\n")
            answer_parts.append("\n".join(rag_lines))
        elif rag_err:
            answer_parts.append(f"⚠️ IMD document search unavailable: {rag_err}")

        answer="\n\n".join(answer_parts)

        return {
            "success":predicted_temp is not None or len(results)>0,
            "tool":"gru_rag",
            "type":"gru_prediction",
            "message":message,
            "prediction":prediction,
            "predicted_temperature_celsius":predicted_temp,
            "sources":sources,
            "answer":answer
        }


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

            if days_ahead==1:
                return {
                    "success":False,
                    "tool":"weather",
                    "type":"tomorrow_forecast",
                    "message":message,
                    "forecast":None,
                    "error":str(e),
                    "answer":f"⚠️ Could not retrieve tomorrow's forecast for {city}: {str(e)}"
                }

            return {
                "success":False,
                "tool":"weather",
                "type":"city_weather",
                "message":message,
                "weather":None,
                "weather_data":None,
                "rag_sources":[],
                "error":str(e),
                "answer":f"⚠️ Could not retrieve current weather for {city}: {str(e)}"
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

                "type":"gru_prediction",

                "message":message,

                "prediction":None,

                "predicted_temperature_celsius":None,

                "error":str(e),

                "answer":f"⚠️ Could not compute GRU next-hour prediction: {str(e)}"

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

                "type":"rag_response",

                "message":message,

                "sources":[],

                "error":str(e),

                "answer":f"⚠️ Could not search IMD documents: {str(e)}"

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
