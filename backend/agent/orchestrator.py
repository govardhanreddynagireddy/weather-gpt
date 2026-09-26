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

from backend.tools.weather_tool import (
    get_weather,
    get_weather_forecast
)

from backend.services.gemini_service import generate_grounded_response
from backend.services.risk_engine import calculate_risk
from backend.services.impact_advisory import generate_advisory
from backend.services.activity_advisor import evaluate_activity_suitability
from backend.tools.historical_tool import compare_weather
from backend.agent.intent_classifier import (
    classify_user_intent,
    is_conversational_intent,
    get_conversational_response,
    INTENT_RAIN_QUERY
)


# =====================================================
# TELUGU CITY MAPPING
# =====================================================

TELUGU_CITY_MAP = {
    "కర్నూలు": "Kurnool",
    "కర్నూలులో": "Kurnool",
    "కడప": "Kadapa",
    "కడపలో": "Kadapa",
    "హైదరాబాద్": "Hyderabad",
    "హైదరాబాద్‌లో": "Hyderabad",
    "విజయవాడ": "Vijayawada",
    "విజయవాడలో": "Vijayawada",
    "తిరుపతి": "Tirupati",
    "తిరుపతిలో": "Tirupati",
    "విశాఖపట్నం": "Visakhapatnam",
    "విశాఖపట్నంలో": "Visakhapatnam",
    "అనంతపురం": "Anantapur",
    "అనంతపురంలో": "Anantapur",
    "గుంటూరు": "Guntur",
    "గుంటూరులో": "Guntur",
    "నెల్లూరు": "Nellore",
    "నెల్లూరులో": "Nellore",
    "వరంగల్": "Warangal",
    "వరంగల్‌లో": "Warangal"
}


from backend.agent.semantic_parser import parse_semantic_intent

# =====================================================
# EXTRACT CITY
# =====================================================

def extract_city(message, location_context=None, conversation_history=None):
    if not message:
        return (location_context or {}).get("city") or "Kurnool"
    info = parse_semantic_intent(message, location_context, conversation_history)
    return info.get("location") or "Kurnool"



def _generate_action_pills(tool, city, days_ahead=0, ui_action=None):
    pills = [
        {"label": f"📍 View {city} on Map", "action": "focus_map", "city": city},
        {"label": "🌧️ Rain Layer", "action": "set_layer", "layer": "precipitation"},
        {"label": "📊 Hourly Forecast", "action": "switch_tab", "tab": "forecast"}
    ]
    if tool == "risk":
        pills.append({"label": "⚠️ Risk Advisory", "action": "switch_tab", "tab": "alerts"})
    elif tool == "rag":
        pills.append({"label": "📄 IMD Bulletins", "action": "switch_tab", "tab": "imd"})
    elif tool in ["gru", "weather_gru", "gru_rag"]:
        pills.append({"label": "📈 WeatherGRU Chart", "action": "switch_tab", "tab": "gru"})
    return pills


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

def orchestrate(message, language="en", location_context=None, conversation_history=None):

    print()
    print("========================================")
    print("ORCHESTRATOR START")
    print("MESSAGE:", message)
    print("LANGUAGE:", language)
    print("LOCATION CTX:", location_context)
    print("CONVERSATION TURNS:", len(conversation_history) if conversation_history else 0)
    print("========================================")


    # =================================================
    # VALIDATE
    # =================================================

    if not message or not message.strip():

        return {
            "success": False,
            "error": "Message cannot be empty.",
            "answer": "⚠️ Message cannot be empty." if language == "en" else "⚠️ సందేశం ఖాళీగా ఉండకూడదు."
        }

    message = message.strip()

    # =================================================
    # SEMANTIC INTENT & LOCATION EXTRACTION
    # =================================================

    semantic_info = parse_semantic_intent(message, location_context, conversation_history)
    city = semantic_info["location"]
    days_ahead = semantic_info["days_ahead"]
    ui_action = semantic_info["ui_action"]
    user_intent = semantic_info.get("intent") or classify_user_intent(message, conversation_history)

    # =================================================
    # CONVERSATIONAL INTENTS (GREETINGS, THANKS, ETC.)
    # MUST NOT EXECUTE ANY WEATHER TOOLS OR REPEAT PREVIOUS FORECASTS
    # =================================================
    if is_conversational_intent(user_intent):
        print(f"DEBUG CONVERSATIONAL INTENT: '{user_intent}' -> Returning conversational response without weather tools.")
        conv_answer = get_conversational_response(user_intent, language=language)

        conv_pills = [
            {"label": f"📍 Weather in {city}", "action": "focus_map", "city": city},
            {"label": "🌧️ Rain Check", "action": "set_layer", "layer": "precipitation"},
            {"label": "🔮 Tomorrow Forecast", "action": "switch_tab", "tab": "forecast"}
        ]

        return {
            "success": True,
            "tool": "conversational",
            "type": "conversational",
            "intent": user_intent,
            "message": message,
            "language": language,
            "answer": conv_answer,
            "action_pills": conv_pills,
            "ui_action": None,
            "location_context": {"city": city, "days_ahead": days_ahead}
        }

    # =================================================
    # ROUTING
    # =================================================

    print("DEBUG 0: ROUTING TOOL")

    tool = route_tool(message)

    print(
        "DEBUG 0: SELECTED TOOL =",
        tool
    )

    action_pills = _generate_action_pills(tool, city, days_ahead, ui_action)

    # =================================================
    # ACTIVITY & PURPOSE ADVISORY (General-Purpose Conversational Weather Intelligence)
    # =================================================
    activity = semantic_info.get("activity")
    if activity:
        print(f"DEBUG ACTIVITY: Detected activity '{activity.get('key')}' in domain '{activity.get('domain')}'")

        current_weather = None
        weather_err = None
        try:
            current_weather = get_weather(city)
        except Exception as e:
            weather_err = str(e)

        forecast_summary = None
        raw_forecast = None
        time_intent = semantic_info.get("time_intent", "current")
        if days_ahead == 1 or time_intent == "tomorrow":
            try:
                raw_forecast = get_weather_forecast(city, days_ahead=1)
                if raw_forecast:
                    forecast_summary = summarize_forecast(city, raw_forecast)
            except Exception:
                pass

        historical_data = {}
        try:
            historical_data = compare_weather(city, current_weather or {})
        except Exception:
            pass

        risk_data = {}
        try:
            risk_data = calculate_risk(current_weather or {}, historical_data)
        except Exception:
            pass

        imd_results = []
        try:
            imd_results = search_weather(f"{city} {activity.get('name')} warning bulletin")
            if not imd_results:
                imd_results = search_weather(f"{city} weather warning")
        except Exception:
            pass

        activity_eval = evaluate_activity_suitability(
            activity=activity,
            weather_data=current_weather,
            forecast_summary=forecast_summary,
            risk_data=risk_data,
            imd_warnings=imd_results,
            time_intent=time_intent,
            city=city,
            language=language
        )

        answer = activity_eval["answer"]

        # Gemini NLG Grounding
        structured_context = {
            "tool": "activity_advisory",
            "activity_name": activity.get("name"),
            "domain": activity.get("domain"),
            "decision_context": activity.get("decision_context"),
            "suitability": activity_eval["suitability"],
            "city": city,
            "time_intent": time_intent,
            "observations": current_weather,
            "forecast": forecast_summary,
            "evaluated_parameters": activity_eval["parameters"],
            "key_factors": activity_eval["key_factors"],
            "recommendations": activity_eval["recommendations"],
            "attributions": activity_eval["attributions"],
            "official_warnings": activity_eval["attributions"]["official_warnings"],
            "direct_answer": activity_eval["direct_answer"]
        }

        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        act_pills = [
            {"label": f"📍 View {city} on Map", "action": "focus_map", "city": city},
            {"label": "🌧️ Rain Layer", "action": "set_layer", "layer": "precipitation"},
            {"label": "⚠️ Risk & Advisories", "action": "switch_tab", "tab": "alerts"}
        ]
        if days_ahead == 1:
            act_pills.append({"label": "📊 Hourly Forecast", "action": "switch_tab", "tab": "forecast"})

        effective_tool = "risk" if (tool == "risk" or activity.get("domain") == "agriculture") else "activity_advisory"

        return {
            "success": True,
            "tool": effective_tool,
            "type": "activity_advisory",
            "activity": activity.get("key"),
            "domain": activity.get("domain"),
            "suitability": activity_eval["suitability"],
            "message": message,
            "language": language,
            "weather": current_weather,
            "forecast": forecast_summary,
            "risk": risk_data,
            "activity_advisory": activity_eval,
            "answer": answer,
            "action_pills": act_pills,
            "ui_action": ui_action,
            "location_context": {"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # COMBINED: WEATHER + GRU
    # =================================================

    if tool=="weather_gru":

        city = semantic_info["location"] or extract_city(message, location_context, conversation_history)
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

        # Gemini NLG Grounding
        structured_context = {
            "tool": "weather_gru",
            "city": city,
            "current_weather": weather_data,
            "gru_prediction": {
                "predicted_temperature_celsius": predicted_temp,
                "horizon": "next_hour",
                "model": "WeatherGRU"
            } if predicted_temp is not None else None,
            "weather_error": weather_err,
            "gru_error": gru_err
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":weather_data is not None or predicted_temp is not None,
            "tool":"weather_gru",
            "type":"city_weather",
            "message":message,
            "language":language,
            "weather":weather_data,
            "weather_data":weather_data,
            "prediction":prediction,
            "predicted_temperature_celsius":predicted_temp,
            "rag_sources":[],
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # COMBINED: WEATHER + RAG
    # =================================================

    if tool=="weather_rag":

        city = semantic_info["location"] or extract_city(message, location_context, conversation_history)
        days_ahead = semantic_info["days_ahead"]
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

        # Gemini NLG Grounding
        structured_context = {
            "tool": "weather_rag",
            "city": city,
            "days_ahead": days_ahead,
            "weather": weather_data if days_ahead==0 else None,
            "forecast": summarize_forecast(city,weather_data) if (days_ahead==1 and weather_data) else None,
            "imd_bulletins": results[:3] if results else [],
            "weather_error": weather_err,
            "rag_error": rag_err
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":weather_data is not None or len(results)>0,
            "tool":"weather_rag",
            "type":"city_weather" if days_ahead==0 else "tomorrow_forecast",
            "message":message,
            "language":language,
            "weather":weather_data if days_ahead==0 else None,
            "forecast":summarize_forecast(city,weather_data) if (days_ahead==1 and weather_data) else None,
            "sources":sources,
            "rag_sources":sources,
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
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

        # Gemini NLG Grounding
        structured_context = {
            "tool": "gru_rag",
            "gru_prediction": {
                "predicted_temperature_celsius": predicted_temp,
                "horizon": "next_hour",
                "model": "WeatherGRU"
            } if predicted_temp is not None else None,
            "imd_bulletins": results[:3] if results else [],
            "gru_error": gru_err,
            "rag_error": rag_err
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":predicted_temp is not None or len(results)>0,
            "tool":"gru_rag",
            "type":"gru_prediction",
            "message":message,
            "language":language,
            "prediction":prediction,
            "predicted_temperature_celsius":predicted_temp,
            "sources":sources,
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # WEATHER
    # =================================================

    if tool=="weather":

        print()
        print("========================================")
        print("WEATHER DEBUG")
        print("========================================")

        if not city:
            city = extract_city(message, location_context)

        print(
            "DEBUG WEATHER 1: CITY =",
            city
        )

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
                    "language":language,
                    "forecast":None,
                    "error":str(e),
                    "answer":f"⚠️ Could not retrieve tomorrow's forecast for {city}: {str(e)}" if language == "en" else f"⚠️ {city} కోసం రేపటి వాతావరణ సమాచారం పొందలేకపోయాము: {str(e)}"
                }

            return {
                "success":False,
                "tool":"weather",
                "type":"city_weather",
                "message":message,
                "language":language,
                "weather":None,
                "weather_data":None,
                "rag_sources":[],
                "error":str(e),
                "answer":f"⚠️ Could not retrieve current weather for {city}: {str(e)}" if language == "en" else f"⚠️ {city} కోసం ప్రస్తుత వాతావరణ సమాచారం పొందలేకపోయాము: {str(e)}"
            }

        print(
            "DEBUG WEATHER 4: WEATHER API DONE"
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

            if user_intent == INTENT_RAIN_QUERY:
                precip = float(rainfall) if rainfall not in ["--", None] else 0.0
                if language == "te":
                    answer = (
                        f"🌧️ {city} లో వర్షపాతం సమాచారం:\n\n"
                        f"వర్షపాతం: {rainfall} mm\n"
                        f"వాతావరణ పరిస్థితి: {condition}\n"
                        f"పరిస్థితి: {'ప్రస్తుతం వర్షం పడుతోంది.' if precip > 0 else 'ప్రస్తుతం వర్షం లేదు.'}"
                    )
                else:
                    answer = (
                        f"🌧️ Rain Information for {city}:\n\n"
                        f"• Precipitation: {rainfall} mm\n"
                        f"• Sky Condition: {condition}\n"
                        f"• Status: {'Rain is currently observed.' if precip > 0 else 'No rain is currently observed.'}"
                    )
            elif language == "te":
                answer=(
                    f"{city} లో ప్రస్తుత వాతావరణం:\n\n"
                    f"🌡️ ఉష్ణోగ్రత: {temperature} °C\n"
                    f"💧 తేమ: {humidity}%\n"
                    f"💨 గాలి వేగం: {wind} km/h\n"
                    f"🌧️ వర్షపాతం: {rainfall} mm\n"
                    f"🌤️ పరిస్థితి: {condition}"
                )
            else:
                answer=(
                    f"Current weather in {city}:\n\n"
                    f"🌡️ Temperature: {temperature} °C\n"
                    f"💧 Humidity: {humidity}%\n"
                    f"💨 Wind: {wind} km/h\n"
                    f"🌧️ Rain: {rainfall} mm\n"
                    f"🌤️ Condition: {condition}"
                )

            # Gemini NLG Grounding
            structured_context = {
                "tool": "weather",
                "intent": user_intent,
                "city": city,
                "current_weather": weather_data
            }
            gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
            if gemini_answer:
                answer = gemini_answer

            return {
                "success":True,
                "tool":"weather",
                "type":"city_weather",
                "intent":user_intent,
                "message":message,
                "language":language,
                "weather":weather_data,
                "weather_data":weather_data,
                "rag_sources":[],
                "answer":answer,
                "action_pills":action_pills,
                "ui_action":ui_action,
                "location_context":{"city": city, "days_ahead": days_ahead}
            }

        # =================================================
        # TOMORROW FORECAST
        # =================================================

        forecast_summary=summarize_forecast(
            city,
            weather_data
        )

        if user_intent == INTENT_RAIN_QUERY:
            precip = float(forecast_summary.get("precipitation_mm", 0.0) or 0.0)
            cond = forecast_summary.get("weather_description", "Partly Cloudy")
            dt = forecast_summary.get("date", "tomorrow")
            if language == "te":
                answer = (
                    f"🌧️ {city} లో రేపటి ({dt}) వర్షపాతం సమాచారం:\n\n"
                    f"అంచనా వేసిన వర్షపాతం: {precip} mm\n"
                    f"వాతావరణ పరిస్థితి: {cond}\n"
                    f"పరిస్థితి: {'రేపు వర్షం పడే అవకాశం ఉంది.' if precip > 0 else 'రేపు వర్షం పడే సూచనలు లేవు.'}"
                )
            else:
                answer = (
                    f"🌧️ Rain Information for {city} tomorrow ({dt}):\n\n"
                    f"• Expected Precipitation: {precip} mm\n"
                    f"• Sky Condition: {cond}\n"
                    f"• Status: {'Rain is expected tomorrow.' if precip > 0 else 'No significant rain is expected tomorrow.'}"
                )
        elif language == "te":
            answer=(
                f"{city} లో రేపటి వాతావరణ అంచనా "
                f"({forecast_summary['date']}):\n\n"
                f"🌡️ గరిష్ట ఉష్ణోగ్రత: {forecast_summary['max_temperature_celsius']} °C\n"
                f"🌡️ కనిష్ట ఉష్ణోగ్రత: {forecast_summary['min_temperature_celsius']} °C\n"
                f"🌧️ వర్షపాతం: {forecast_summary['precipitation_mm']} mm\n"
                f"🌤️ పరిస్థితి: {forecast_summary['weather_description']}"
            )
        else:
            answer=(
                f"Tomorrow's forecast for {city} "
                f"({forecast_summary['date']}):\n\n"
                f"🌡️ Max: {forecast_summary['max_temperature_celsius']} °C\n"
                f"🌡️ Min: {forecast_summary['min_temperature_celsius']} °C\n"
                f"🌧️ Precipitation: {forecast_summary['precipitation_mm']} mm\n"
                f"🌤️ Condition: {forecast_summary['weather_description']}"
            )

        # Gemini NLG Grounding
        structured_context = {
            "tool": "weather_forecast",
            "intent": user_intent,
            "city": city,
            "forecast": forecast_summary
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":True,
            "tool":"weather",
            "type":"tomorrow_forecast",
            "message":message,
            "language":language,
            "forecast":forecast_summary,
            "raw_forecast":weather_data,
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # GRU → NEXT HOUR
    # =================================================

    if tool=="gru":

        print()
        print("========================================")
        print("GRU DEBUG")
        print("========================================")

        try:
            prediction=predict_latest_temperature()
        except Exception as e:
            return {
                "success":False,
                "tool":"gru",
                "type":"gru_prediction",
                "message":message,
                "language":language,
                "prediction":None,
                "predicted_temperature_celsius":None,
                "error":str(e),
                "answer":f"⚠️ Could not compute GRU next-hour prediction: {str(e)}"
            }

        model_info=get_gru_info()

        predicted_temperature=prediction
        if isinstance(prediction, dict):
            predicted_temperature=prediction.get(
                "prediction_celsius",
                prediction.get(
                    "predicted_temperature_celsius",
                    prediction.get("prediction", prediction)
                )
            )

        if language == "te":
            answer=(
                "📈 తరువాతి గంట ఉష్ణోగ్రత అంచనా (WeatherGRU):\n\n"
                f"మోడల్ అంచనా ప్రకారం తరువాతి గంటలో ఉష్ణోగ్రత సుమారు "
                f"{predicted_temperature:.2f} °C ఉండవచ్చు."
            )
        else:
            answer=(
                "📈 Next-hour temperature prediction:\n\n"
                f"The WeatherGRU model predicts the "
                f"temperature to be approximately "
                f"{predicted_temperature:.2f} °C."
            )

        # Gemini NLG Grounding
        structured_context = {
            "tool": "gru",
            "predicted_temperature_celsius": predicted_temperature,
            "horizon": "next_hour",
            "model": model_info
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":True,
            "tool":"gru",
            "type":"gru_prediction",
            "message":message,
            "language":language,
            "prediction":prediction,
            "predicted_temperature_celsius":predicted_temperature,
            "model":model_info,
            "source":"era5_merged.nc",
            "prediction_horizon":"next_hour",
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # RAG → FAISS
    # =================================================

    if tool=="rag":

        print()
        print("========================================")
        print("RAG DEBUG")
        print("========================================")

        try:
            results=search_weather(message)
        except Exception as e:
            return {
                "success":False,
                "tool":"rag",
                "type":"rag_response",
                "message":message,
                "language":language,
                "sources":[],
                "error":str(e),
                "answer":f"⚠️ Could not search IMD documents: {str(e)}"
            }

        if not results:
            no_info_ans = (
                "No relevant information was found in the IMD documents."
                if language == "en" else
                "IMD పత్రాలలో ఎటువంటి సంబంధిత సమాచారం కనుగొనబడలేదు."
            )
            return {
                "success":True,
                "tool":"rag",
                "type":"rag_response",
                "message":message,
                "language":language,
                "answer":no_info_ans,
                "sources":[]
            }

        answer="Relevant IMD information:\n\n"
        for i,result in enumerate(results, 1):
            answer+=(
                f"Source {i}:\n"
                f"{result['text']}\n\n"
            )

        sources=[]
        for result in results:
            sources.append({
                "source":result["source"],
                "chunk_id":result["chunk_id"],
                "score":result["score"]
            })

        # Gemini NLG Grounding
        structured_context = {
            "tool": "rag",
            "query": message,
            "imd_bulletins": results[:4]
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":True,
            "tool":"rag",
            "type":"rag_response",
            "message":message,
            "language":language,
            "answer":answer,
            "sources":sources,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # HISTORICAL
    # =================================================

    if tool=="historical":

        city = semantic_info["location"] or extract_city(message, location_context, conversation_history)
        weather_data=None
        try:
            weather_data=get_weather(city)
        except Exception:
            pass

        historical_data=compare_weather(city, weather_data or {})

        if historical_data.get("available"):
            comp=historical_data["comparison"]
            t_curr=comp["temperature"]["current"]
            t_avg=comp["temperature"]["historical_average"]
            t_diff=comp["temperature"]["difference"]
            t_anom=comp["temperature"]["anomaly_percent"]
            h_curr=comp["humidity"]["current"]
            h_avg=comp["humidity"]["historical_average"]
            w_curr=comp["wind_speed"]["current"]
            w_avg=comp["wind_speed"]["historical_average"]
            sign="+" if t_diff >= 0 else ""

            if language == "te":
                answer=(
                    f"📊 {city} చారిత్రక వాతావరణ పోలిక:\n\n"
                    f"🌡️ ప్రస్తుత ఉష్ణోగ్రత: {t_curr} °C (చారిత్రక సగటు: {t_avg} °C, తేడా: {sign}{t_diff} °C, వ్యత్యాసం: {t_anom}%)\n"
                    f"💧 తేమ: {h_curr}% (చారిత్రక సగటు: {h_avg}%)\n"
                    f"💨 గాలి వేగం: {w_curr} km/h (చారిత్రక సగటు: {w_avg} km/h)"
                )
            else:
                answer=(
                    f"📊 Historical Weather Comparison for {city}:\n\n"
                    f"🌡️ Current Temp: {t_curr} °C (Historical Avg: {t_avg} °C, Diff: {sign}{t_diff} °C, Anomaly: {t_anom}%)\n"
                    f"💧 Humidity: {h_curr}% (Historical Avg: {h_avg}%)\n"
                    f"💨 Wind Speed: {w_curr} km/h (Historical Avg: {w_avg} km/h)"
                )
        else:
            answer=(
                f"⚠️ Historical data is not available for {city}."
                if language == "en" else
                f"⚠️ {city} కోసం చారిత్రక వాతావరణ సమాచారం అందుబాటులో లేదు."
            )

        structured_context = {
            "tool": "historical",
            "city": city,
            "current_weather": weather_data,
            "historical_comparison": historical_data
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":True,
            "tool":"historical",
            "type":"historical_analysis",
            "message":message,
            "language":language,
            "weather":weather_data,
            "historical":historical_data,
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # RISK
    # =================================================

    if tool=="risk":

        city = semantic_info["location"] or extract_city(message, location_context, conversation_history)
        weather_data=None
        try:
            weather_data=get_weather(city)
        except Exception:
            pass

        historical_data=compare_weather(city, weather_data or {})
        risk_data=calculate_risk(weather_data or {}, historical_data)
        advisory_data=generate_advisory(weather_data or {}, risk_data)

        level=risk_data.get("risk_level", "LOW")
        score=risk_data.get("risk_score", 0)
        reasons_str=", ".join(risk_data.get("reasons", [])) or ("Normal conditions" if language == "en" else "సాధారణ పరిస్థితులు")
        impacts_str="\n".join(f"• {imp}" for imp in advisory_data.get("possible_impacts", []))
        recs_str="\n".join(f"• {rec}" for rec in advisory_data.get("recommendations", []))

        if language == "te":
            answer=(
                f"⚠️ {city} వాతావరణ ప్రమాద అంచనా & సలహా:\n\n"
                f"ప్రమాద స్థాయి: {level} (స్కోర్: {score}/100)\n"
                f"ప్రధాన కారణాలు: {reasons_str}\n\n"
                f"సాధ్యమయ్యే ప్రభావాలు:\n{impacts_str}\n\n"
                f"సిఫార్సులు:\n{recs_str}"
            )
        else:
            answer=(
                f"⚠️ Weather Risk Analysis & Advisory for {city}:\n\n"
                f"Risk Level: {level} (Score: {score}/100)\n"
                f"Key Factors: {reasons_str}\n\n"
                f"Possible Impacts:\n{impacts_str}\n\n"
                f"Recommendations:\n{recs_str}"
            )

        structured_context = {
            "tool": "risk",
            "city": city,
            "current_weather": weather_data,
            "risk": risk_data,
            "advisory": advisory_data,
            "historical": historical_data
        }
        gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
        if gemini_answer:
            answer = gemini_answer

        return {
            "success":True,
            "tool":"risk",
            "type":"risk_advisory",
            "message":message,
            "language":language,
            "weather":weather_data,
            "risk":risk_data,
            "advisory":advisory_data,
            "historical":historical_data,
            "answer":answer,
            "action_pills":action_pills,
            "ui_action":ui_action,
            "location_context":{"city": city, "days_ahead": days_ahead}
        }


    # =================================================
    # FALLBACK
    # =================================================

    print(
        "DEBUG FALLBACK: NO TOOL MATCHED"
    )

    if language == "te":
        fallback_ans=(
            "నేను ఈ క్రింది విషయాలలో సహాయం చేయగలను:\n\n"
            "🌤️ ప్రస్తుత వాతావరణం\n"
            "🔮 రేపటి వాతావరణ అంచనా\n"
            "📈 తరువాతి గంట ఉష్ణోగ్రత అంచనా (WeatherGRU)\n"
            "⚠️ వాతావరణ ప్రమాదం & వ్యవసాయ సలహాలు\n"
            "📄 IMD అధికారిక సమాచారం & హెచ్చరికలు"
        )
    else:
        fallback_ans=(
            "I can help with:\n\n"
            "🌤️ Current weather\n"
            "🔮 Tomorrow's forecast\n"
            "📈 Next-hour temperature prediction (WeatherGRU)\n"
            "⚠️ Weather risk & agricultural advisories\n"
            "📄 IMD weather bulletins & warnings"
        )

    structured_context = {
        "tool": "fallback",
        "query": message,
        "capabilities": [
            "Current weather observation",
            "Tomorrow's forecast",
            "Next-hour WeatherGRU temperature prediction",
            "Weather risk analysis & impact advisory",
            "Official IMD bulletins and warnings via RAG"
        ]
    }
    gemini_answer = generate_grounded_response(message, structured_context, language, conversation_history)
    if gemini_answer:
        fallback_ans = gemini_answer

    return {
        "success":True,
        "tool":"fallback",
        "type":"fallback_response",
        "message":message,
        "language":language,
        "answer":fallback_ans,
        "action_pills":action_pills,
        "ui_action":ui_action,
        "location_context":{"city": city, "days_ahead": days_ahead}
    }
