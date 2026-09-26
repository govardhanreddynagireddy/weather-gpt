"""
Intent Classification Module for WeatherGPT+
Classifies user messages independently before selecting or executing weather tools.
Guarantees that conversational intents (greetings, goodbyes, thanks, acknowledgements)
never trigger heavy meteorological tools or cause previous weather forecasts to repeat.
"""

import re
import unicodedata
from typing import Optional, Dict, Any, List

# Standard Intent Constants
INTENT_GREETING = "greeting"
INTENT_GOODBYE = "goodbye"
INTENT_THANKS = "thanks"
INTENT_ACKNOWLEDGEMENT = "acknowledgement"
INTENT_WEATHER_CURRENT = "weather_current"
INTENT_WEATHER_FORECAST = "weather_forecast"
INTENT_RAIN_QUERY = "rain_query"
INTENT_RISK_SAFETY = "risk_safety"
INTENT_FARMING_ADVISORY = "farming_advisory"
INTENT_HISTORICAL = "historical"
INTENT_IMD_RAG = "imd_rag"
INTENT_MAP_LAYER = "map_layer"
INTENT_LOCATION_CHANGE = "location_change"
INTENT_WEATHER_FOLLOWUP = "general_weather_followup"
INTENT_UNKNOWN = "unknown_general"

CONVERSATIONAL_INTENTS = {
    INTENT_GREETING,
    INTENT_GOODBYE,
    INTENT_THANKS,
    INTENT_ACKNOWLEDGEMENT
}

# Regex for Pure Conversational Messages (must not contain weather questions)
GREETING_REGEX = re.compile(
    r"^(\s*(hi|hello|hey|heyy|heya|howdy|yo|sup|namaste|namaskar|namaskaram|greetings|hola)\b[!?. ]*|"
    r"\s*(good\s+(morning|afternoon|evening|day))\b[!?. ]*|"
    r"\s*(హలో|హాయ్|నమస్కారం|నమస్తే|బాగున్నారా|నమస్కారాలు)[!?. ]*)$",
    re.IGNORECASE
)

GOODBYE_REGEX = re.compile(
    r"^(\s*(bye|goodbye|see\s+ya|see\s+you|cya|take\s+care|farewell|good\s*night)\b[!?. ]*|"
    r"\s*(టాటా|బై|వెళ్లొస్తా|వెళ్లొస్తాను|మళ్ళీ\s+కలుద్దాం)[!?. ]*)$",
    re.IGNORECASE
)

THANKS_REGEX = re.compile(
    r"^(\s*(thanks|thank\s+you|thankyou|thx|many\s+thanks|appreciate\s+it|cheers)\b[!?. ]*|"
    r"\s*(ధన్యవాదాలు|థాంక్స్|చాలా\s+ధన్యవాదాలు|ధన్యవాదములు|thanks\s+a\s+lot)[!?. ]*)$",
    re.IGNORECASE
)

ACKNOWLEDGEMENT_REGEX = re.compile(
    r"^(\s*(ok|okay|got\s+it|fine|cool|understood|all\s+right|alright|great|nice|sure|k|sounds\s+good)\b[!?. ]*|"
    r"\s*(సరే|అర్థమైంది|మంచిది|ఓకే|సరేనండి)[!?. ]*)$",
    re.IGNORECASE
)

# Weather Keywords to prevent false positives when greeting is mixed with a question
WEATHER_KEYWORDS = [
    "weather", "forecast", "temperature", "temp", "rain", "rainfall", "precipitation",
    "wind", "cloud", "humidity", "repu", "tomorrow", "today", "yesterday", "safe",
    "farming", "spray", "crop", "imd", "warning", "bulletin", "వాతావరణం", "వర్షం",
    "ఎండ", "చలి", "తుఫాను", "హెచ్చరిక"
]


def normalize_text(text: str) -> str:
    """Normalizes accents on Latin characters without stripping Indic vowel signs."""
    if not text:
        return ""
    norm = unicodedata.normalize('NFKD', str(text))
    cleaned = ''.join(c for c in norm if not (unicodedata.combining(c) and ord(c) < 0x0900))
    return cleaned.strip()


def classify_user_intent(message: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Independently classifies the user's message intent before any weather tool is executed.
    Guarantees greetings and conversational messages are separated from meteorological inquiries.
    """
    if not message or not str(message).strip():
        return INTENT_UNKNOWN

    raw = str(message).strip()
    norm = normalize_text(raw)
    msg_lower = " " + " ".join(norm.lower().split()) + " "

    # 1. PURE CONVERSATIONAL INTENTS (Top Priority)
    # Check if message contains weather keywords
    has_weather_keyword = any(kw in msg_lower for kw in WEATHER_KEYWORDS)

    if not has_weather_keyword:
        if GREETING_REGEX.match(raw) or GREETING_REGEX.match(norm):
            return INTENT_GREETING

        if GOODBYE_REGEX.match(raw) or GOODBYE_REGEX.match(norm):
            return INTENT_GOODBYE

        if THANKS_REGEX.match(raw) or THANKS_REGEX.match(norm):
            return INTENT_THANKS

        if ACKNOWLEDGEMENT_REGEX.match(raw) or ACKNOWLEDGEMENT_REGEX.match(norm):
            return INTENT_ACKNOWLEDGEMENT

    # 2. MAP / LAYER ACTIONS
    map_tokens = [
        "show rain", "rain map", "precipitation layer", "show wind", "wind map",
        "show clouds", "cloud map", "show temperature", "temp map", "show risk",
        "risk map", "వర్షం మ్యాప్", "గాలి మ్యాప్", "మేఘాల మ్యాప్"
    ]
    if any(tok in msg_lower for tok in map_tokens):
        return INTENT_MAP_LAYER

    # 3. RAIN & PRECIPITATION QUERIES
    rain_tokens = [
        "will it rain", "is it raining", "is it going to rain", "rain forecast",
        "chance of rain", "any rain", "precipitation", "rain?", "and rain?",
        "does it rain", "expecting rain", "वर्षा", "వర్షం", "వర్షపాతం",
        "వర్షం పడుతుందా", "varsham paduthunda", "varsham untada", "varsham"
    ]
    # Check if query is primarily about rain
    if any(tok in msg_lower for tok in rain_tokens) or (msg_lower.strip() in ["rain", "rain?", "varsham", "varsham?"]):
        return INTENT_RAIN_QUERY

    # 4. LOCATION CHANGE ("What about Hyderabad?", "how about Kurnool?", "in Vijayawada?")
    # Check if a new city is mentioned with a transition phrase
    location_change_indicators = ["what about ", "how about ", "and in ", "what is in ", "weather in "]
    from backend.agent.semantic_parser import KNOWN_CITIES
    has_city_mention = any(c.lower() in msg_lower for c in KNOWN_CITIES)
    has_explicit_tomorrow = any(tok in msg_lower for tok in ["tomorrow", "tommorow", "tommorrow", "tomorow", "repu", "రేపు"])
    if not has_explicit_tomorrow and has_city_mention and any(ind in msg_lower for ind in ["what about ", "how about ", "and in "]):
        return INTENT_LOCATION_CHANGE

    # 5. FARMING & AGRICULTURAL ADVISORY
    farming_tokens = [
        "safe for farming", "farming", "crop", "crops", "pesticide", "spray",
        "spraying", "fertilizer", "harvest", "harvesting", "dry crop", "irrigation",
        "weedicide", "రైతు", "వ్యవసాయం", "పంట", "స్ప్రే", "మందు కొట్ట"
    ]
    if any(tok in msg_lower for tok in farming_tokens):
        return INTENT_FARMING_ADVISORY

    # 6. WEATHER FORECAST / TOMORROW
    forecast_tokens = [
        "tomorrow", "tommorow", "tommorrow", "tomorow", "tommrow", "repu", "next day", "repatiki", "forecast", "future weather",
        "tomorrow's weather", "what about tomorrow", "weather tomorrow", "రేపు",
        "రేపటి వాతావరణం", "రేపటి అంచనా", "repu ela", "repu weather ela untadhi", "repu weather",
        "రేపు వాతావరణం ఎలా ఉంటుంది", "వాతావరణం ఎలా ఉంటుంది"
    ]
    if any(tok in msg_lower for tok in forecast_tokens):
        return INTENT_WEATHER_FORECAST

    # 7. RISK & SAFETY
    risk_tokens = [
        "risk", "danger", "dangerous", "is it safe", "safe?", "safety",
        "severe", "warning", "alert", "threat", "ప్రమాదం", "రిస్క్", "సురక్షితమా"
    ]
    if any(tok in msg_lower for tok in risk_tokens):
        return INTENT_RISK_SAFETY

    # 8. HISTORICAL WEATHER
    hist_tokens = [
        "historical", "history", "last year", "previous year", "past weather",
        "past temperature", "earlier years", "చరిత్ర", "గత సంవత్సరం"
    ]
    if any(tok in msg_lower for tok in hist_tokens):
        return INTENT_HISTORICAL

    # 9. IMD / RAG
    imd_tokens = [
        "imd", "bulletin", "official forecast", "imd warning", "cyclone alert",
        "storm warning", "press release", "హెచ్చరిక", "బులెటిన్"
    ]
    if any(tok in msg_lower for tok in imd_tokens):
        return INTENT_IMD_RAG

    # 10. CURRENT WEATHER OBSERVATION
    current_tokens = [
        "current weather", "weather now", "how is the weather", "what is the weather",
        "temperature today", "today's weather", "climate", "temperature", "వాతావరణం",
        "ఉష్ణోగ్రత", "వాతావరణం ఎలా ఉంది", "weather today", "weather here",
        "weather ela undhi", "weather ela untadhi", "weather ela untundhi", "ela undhi", "ela untadhi"
    ]
    if any(tok in msg_lower for tok in current_tokens) or has_city_mention:
        return INTENT_WEATHER_CURRENT

    # 11. GENERAL WEATHER FOLLOWUP (if in conversation)
    followup_tokens = ["humidity", "wind", "wind speed", "pressure", "clouds", "air quality"]
    if any(tok in msg_lower for tok in followup_tokens):
        return INTENT_WEATHER_FOLLOWUP

    return INTENT_UNKNOWN


def is_conversational_intent(intent: str) -> bool:
    """Returns True if the intent is purely conversational and must not call weather tools."""
    return intent in CONVERSATIONAL_INTENTS


def get_conversational_response(intent: str, language: str = "en") -> str:
    """Returns standard friendly conversational responses without running weather APIs."""
    if intent == INTENT_GREETING:
        if language == "te":
            return "నమస్కారం! 👋 నేను WeatherGPT+. నేను వాతావరణ అంచనాలు, వర్షం, ప్రయాణం, వ్యవసాయం, అవుట్‌డోర్ కార్యకలాపాలు, వాతావరణ ప్రమాదాలు, హెచ్చరికలు మొదలైన వాటిలో సహాయపడగలను. మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?"
        return "Hello! 👋 I’m WeatherGPT+. I can help with forecasts, rain, travel, farming, outdoor activities, weather risks, alerts, and more. What would you like to know?"

    if intent == INTENT_GOODBYE:
        if language == "te":
            return "వెళ్లొస్తాను! 👋 జాగ్రత్తగా ఉండండి, మంచి రోజు కావాలని కోరుకుంటున్నాను!"
        return "Goodbye! 👋 Stay safe and have a great day!"

    if intent == INTENT_THANKS:
        if language == "te":
            return "ధన్యవాదాలు! 😊 మీకు మరేదైనా వాతావరణ సమాచారం కావాలంటే అడగండి."
        return "You're welcome! 😊 Let me know if you need anything else."

    if intent == INTENT_ACKNOWLEDGEMENT:
        if language == "te":
            return "సంతోషం! మీకు ఇతర వాతావరణ ప్రశ్నలు ఉంటే తప్పకుండా అడగండి."
        return "Glad to help! Let me know if you have more weather questions."

    return "Hello! How can I assist you with the weather today?"
