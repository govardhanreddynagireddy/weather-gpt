"""
Semantic Natural Language Understanding & Entity Parser for WeatherGPT+
Handles time intent, location extraction (with Telugu/English support & relative context),
and map UI action triggers.
"""

import re
import unicodedata
from typing import Dict, Any, Optional
from backend.agent.activity_registry import find_matching_activity
from backend.agent.intent_classifier import (
    classify_user_intent,
    is_conversational_intent,
    INTENT_LOCATION_CHANGE
)

# Telugu to English City Name Mapping
TELUGU_CITY_MAP = {
    "కర్నూలు": "Kurnool",
    "కర్నూలులో": "Kurnool",
    "కర్నూలుకి": "Kurnool",
    "కడప": "Kadapa",
    "కడపలో": "Kadapa",
    "కడపకి": "Kadapa",
    "హైదరాబాద్": "Hyderabad",
    "హైదరాబాద్‌లో": "Hyderabad",
    "హైదరాబాద్‌కి": "Hyderabad",
    "విజయవాడ": "Vijayawada",
    "విజయవాడలో": "Vijayawada",
    "విజయవాడకి": "Vijayawada",
    "తిరుపతి": "Tirupati",
    "తిరుపతిలో": "Tirupati",
    "తిరుపతికి": "Tirupati",
    "విశాఖపట్నం": "Visakhapatnam",
    "విశాఖపట్నంలో": "Visakhapatnam",
    "విశాఖలో": "Visakhapatnam",
    "వైజాగ్": "Visakhapatnam",
    "వైజాగ్‌లో": "Visakhapatnam",
    "అనంతపురం": "Anantapur",
    "అనంతపురంలో": "Anantapur",
    "గుంటూరు": "Guntur",
    "గుంటూరులో": "Guntur",
    "నెల్లూరు": "Nellore",
    "నెల్లూరులో": "Nellore",
    "వరంగల్": "Warangal",
    "వరంగల్‌లో": "Warangal",
    "బెంగళూరు": "Bengaluru",
    "చెన్నై": "Chennai",
    "ఢిల్లీ": "Delhi",
    "ముంబై": "Mumbai",
    "ఆలంపూర్": "Alampur",
    "ఆలంపూర్‌లో": "Alampur",
    "ఆలంపూర్‌కి": "Alampur"
}

# Common Indian cities for quick lookup
KNOWN_CITIES = [
    "Kurnool", "Hyderabad", "Anantapur", "Vijayawada", "Kadapa",
    "Tirupati", "Visakhapatnam", "Vizag", "Guntur", "Nellore",
    "Warangal", "Bengaluru", "Bangalore", "Chennai", "Madras",
    "Delhi", "Mumbai", "Bombay", "Kolkata", "Calcutta",
    "Adoni", "Nandyal", "Proddatur", "Chittoor", "Rajahmundry",
    "Kakinada", "Eluru", "Ongole", "Machilipatnam", "Tenali",
    "Alampur", "Ālampur"
]

# Words that MUST NEVER be interpreted as city names
NON_CITY_WORDS = {
    "repu", "tomorrow", "today", "yesterday", "now", "here", "near", "me",
    "weather", "forecast", "temperature", "rain", "rainfall", "wind", "clouds",
    "climate", "humidity", "danger", "risk", "alert", "warning", "bulletin",
    "imd", "gru", "hour", "hours", "next", "current", "show", "tell", "what",
    "how", "is", "the", "in", "at", "for", "of", "and", "please", "will",
    "it", "ela", "untadhi", "untundi", "padutunda", "vasthunda", "vachinda",
    "varsham", "yela", "undhi", "undi", "cheppu", "chupinchu", "ippudu",
    "ikkada", "dangerously", "severe", "advisory", "farmer", "farmers",
    "about", "give", "provide", "fetch", "check", "spray", "spraying", "safe",
    "safely", "farming", "crop", "crops", "pesticide", "fertilizer", "field",
    "fields", "can", "could", "should", "would", "any", "rainy", "sunny",
    "cloudy", "hot", "cold", "warm", "tell", "say", "suggest",
    "hi", "hello", "hey", "thanks", "thank", "you", "ok", "okay", "bye", "goodbye",
    "namaste", "morning", "evening", "afternoon", "fine", "cool", "sure", "well",
    "tommorow", "tommorrow", "tomorow", "tommrow", "repatiki", "రేపటి",
    "outside", "out", "doot", "drying", "harvest", "harvesting"
}


def clean_city_token(token: str) -> str:
    """Normalizes accents, diacritics, and strips non-alphabetic chars."""
    if not token:
        return ""
    norm = unicodedata.normalize('NFKD', str(token))
    cleaned = ''.join(c for c in norm if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z]", "", cleaned)
    return cleaned.strip()


def parse_semantic_intent(
    message: str,
    location_context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[list] = None
) -> Dict[str, Any]:
    """
    Parses a user natural language query into structured semantic entities:
    - time_intent: 'tomorrow' | 'next_hour' | 'current' | 'today'
    - days_ahead: 0 or 1
    - location: resolved city name
    - coordinates: {lat, lon} if available from location_context
    - is_relative_location: True if user said 'here' / 'near me'
    - ui_action: Optional map/UI command (e.g. set_layer, geolocate)
    - conversation_history: past turns to maintain multi-turn memory
    """
    if not message or not str(message).strip():
        fallback_city = (location_context or {}).get("city") or "Kurnool"
        return {
            "intent": "unknown_general",
            "time_intent": "current",
            "days_ahead": 0,
            "location": fallback_city,
            "coordinates": None,
            "is_relative_location": False,
            "ui_action": None,
            "activity": None,
            "domain": None,
            "decision_context": None,
            "required_variables": []
        }

    raw_message = str(message).strip()
    msg_lower = " " + " ".join(raw_message.lower().split()) + " "

    # Independent Intent Classification
    user_intent = classify_user_intent(raw_message, conversation_history)

    # 0. ACTIVITY & PURPOSE EXTRACTION (General-Purpose Conversational Weather Intelligence)
    matched_activity = None
    if not is_conversational_intent(user_intent):
        matched_activity = find_matching_activity(raw_message)
        if not matched_activity and conversation_history:
            is_activity_continuation = any(w in msg_lower for w in ["safe", "can i", "can we", "should i", "spray", "dry", "harvest", "umbrella", "suit", "fine to", "repu cheyyocha", "చేయవచ్చా", "సురక్షితమా"])
            if is_activity_continuation:
                for turn in reversed(conversation_history[-4:]):
                    if turn.get("role") == "user":
                        prev_msg = turn.get("content") or turn.get("message") or ""
                        prev_act = find_matching_activity(prev_msg)
                        if prev_act:
                            matched_activity = prev_act
                            break

    # 1. TIME INTENT EXTRACTION
    time_intent = "current"
    days_ahead = 0

    if is_conversational_intent(user_intent):
        # Pure conversational messages NEVER inherit tomorrow or weather forecast
        time_intent = "current"
        days_ahead = 0
    else:
        tomorrow_indicators = [
            "tomorrow", "tommorow", "tommorrow", "tomorow", "tommrow",
            "repu", "రేపు", "next day", "repatiki", "రేపటి", "repu weather ela untadhi"
        ]
        next_hour_indicators = ["next hour", "next-hour", "తరువాతి గంట", "తదుపరి గంట", "next 1 hour"]
        today_indicators = ["today", "now", "current", "ఈరోజు", "ఇప్పుడు", "ప్రస్తుతం"]

        has_explicit_tomorrow = any(ind in msg_lower for ind in tomorrow_indicators)
        has_explicit_next_hour = any(ind in msg_lower for ind in next_hour_indicators)
        has_explicit_today = any(ind in msg_lower for ind in today_indicators)

        if has_explicit_next_hour:
            time_intent = "next_hour"
        elif has_explicit_tomorrow:
            time_intent = "tomorrow"
            days_ahead = 1
        elif user_intent == INTENT_LOCATION_CHANGE:
            # Location change without tomorrow (e.g. "What about Hyderabad?") queries current weather for new city
            time_intent = "current"
            days_ahead = 0
        elif not has_explicit_today and conversation_history:
            # Check if this is an actual weather follow-up query inheriting time horizon
            follow_up_tokens = [
                "rain", "rain?", "varsham", "వర్షం", "safe", "farming", "crop", "crops",
                "spray", "spraying", "pesticide", "wind", "temp", "temperature", "forecast", "danger",
                "outside", "out", "doot", "travel", "driving", "dry", "drying", "harvest", "harvesting"
            ]
            is_weather_followup = any(tok in msg_lower for tok in follow_up_tokens) or (matched_activity is not None)
            if is_weather_followup:
                for turn in reversed(conversation_history[-4:]):
                    if turn.get("role") == "user":
                        prev_text = (turn.get("content") or turn.get("message") or "").lower()
                        if any(ind in prev_text for ind in tomorrow_indicators):
                            time_intent = "tomorrow"
                            days_ahead = 1
                            break
                        elif any(ind in prev_text for ind in next_hour_indicators):
                            time_intent = "next_hour"
                            break

    # 2. UI ACTION EXTRACTION (Chat-to-Map Synchronization)
    ui_action = None

    if any(k in msg_lower for k in ["show rainfall", "rain map", "show rain", "precipitation layer", "వర్షం మ్యాప్", "వర్షపాతం చూపించు"]):
        ui_action = {"type": "set_layer", "layer": "precipitation"}
    elif any(k in msg_lower for k in ["show wind", "wind map", "wind layer", "గాలి మ్యాప్", "గాలి వేగం చూపించు"]):
        ui_action = {"type": "set_layer", "layer": "wind"}
    elif any(k in msg_lower for k in ["show clouds", "cloud map", "cloud layer", "మేఘాల మ్యాప్"]):
        ui_action = {"type": "set_layer", "layer": "clouds"}
    elif any(k in msg_lower for k in ["show temperature", "temp map", "temperature layer", "ఉష్ణోగ్రత మ్యాప్"]):
        ui_action = {"type": "set_layer", "layer": "temperature"}
    elif any(k in msg_lower for k in ["show dangerous areas", "risk map", "show risk", "రిస్క్ మ్యాప్", "ప్రమాదకర ప్రాంతాలు"]):
        ui_action = {"type": "set_layer", "layer": "risk"}
    elif any(k in msg_lower for k in ["weather here", "weather near me", "my location", "ఇక్కడ వాతావరణం", "నా లొకేషన్", "చుట్టుపక్కల"]):
        ui_action = {"type": "geolocate"}

    # 3. RELATIVE LOCATION CHECK
    relative_indicators = ["here", "near me", "around here", "my location", "current location", "ఇక్కడ", "చుట్టుపక్కల", "ఈ ప్రాంతం"]
    is_relative_location = any(rel in msg_lower for rel in relative_indicators)

    # 4. LOCATION EXTRACTION
    resolved_city = None

    # A. Check known Telugu city dictionary
    for telugu_word, eng_city in TELUGU_CITY_MAP.items():
        if telugu_word in raw_message:
            resolved_city = eng_city
            break

    # B. Check known English cities list
    if not resolved_city:
        for known in KNOWN_CITIES:
            pattern = r"\b" + re.escape(known.lower()) + r"\b"
            if re.search(pattern, msg_lower):
                resolved_city = known
                break

    # C. Prefix & Preposition extraction: "in <City>", "to <City>", "what about <City>", "how about <City>"
    if not resolved_city and not is_relative_location:
        for prefix in ["what about ", "how about ", " in ", " at ", " for ", " to ", " of "]:
            if prefix in msg_lower:
                parts = msg_lower.split(prefix, 1)
                candidate_tokens = parts[1].strip().split()
                if candidate_tokens:
                    first_token = clean_city_token(candidate_tokens[0])
                    if first_token and first_token.lower() not in NON_CITY_WORDS and len(first_token) > 2:
                        resolved_city = first_token.title()
                        break

    # D. Multi-turn conversation history lookup (if city not mentioned in current turn)
    if not resolved_city and not is_relative_location and conversation_history:
        for turn in reversed(conversation_history[-6:]):
            prev_text = turn.get("content") or turn.get("message") or ""
            # Check Telugu
            for telugu_word, eng_city in TELUGU_CITY_MAP.items():
                if telugu_word in prev_text:
                    resolved_city = eng_city
                    break
            if resolved_city:
                break
            # Check English known cities
            prev_lower = " " + " ".join(prev_text.lower().split()) + " "
            for known in KNOWN_CITIES:
                pattern = r"\b" + re.escape(known.lower()) + r"\b"
                if re.search(pattern, prev_lower):
                    resolved_city = known
                    break
            if resolved_city:
                break
            # Check if turn contains city metadata
            if isinstance(turn.get("city"), str) and turn.get("city"):
                resolved_city = turn.get("city")
                break

    # E. Fallback to active location context from client if relative location or no city specified
    active_ctx = location_context or {}
    ctx_city = active_ctx.get("city")
    ctx_lat = active_ctx.get("lat")
    ctx_lon = active_ctx.get("lon")

    if not resolved_city:
        resolved_city = ctx_city if (ctx_city and ctx_city.lower() != "undefined") else "Kurnool"

    # Normalize resolved city if necessary
    resolved_city = clean_city_token(resolved_city) or "Kurnool"

    # Coordinates
    coords = None
    if ctx_lat is not None and ctx_lon is not None:
        coords = {"lat": float(ctx_lat), "lon": float(ctx_lon)}

    return {
        "intent": user_intent,
        "time_intent": time_intent,
        "days_ahead": days_ahead,
        "location": resolved_city,
        "coordinates": coords,
        "is_relative_location": is_relative_location,
        "ui_action": ui_action,
        "activity": matched_activity,
        "domain": matched_activity.get("domain") if matched_activity else None,
        "decision_context": matched_activity.get("decision_context") if matched_activity else None,
        "required_variables": matched_activity.get("required_variables", []) if matched_activity else []
    }

