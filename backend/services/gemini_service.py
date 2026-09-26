import os
import json
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger("weathergpt.gemini")

DEFAULT_MODEL = "gemini-3.6-flash"

GROUNDED_SYSTEM_PROMPT = """You are WeatherGPT+, an authoritative and grounded weather intelligence assistant.

Your primary duty is to explain weather conditions, ML predictions, risk assessments, advisories, and official IMD bulletins based STRICTLY on the trusted context provided by the backend.

CRITICAL GROUNDING RULES:
1. Use ONLY the trusted weather context supplied. Never invent, estimate, or extrapolate numerical weather values (temperature, humidity, wind speed, rainfall, pressure).
2. Never fabricate weather warnings, alerts, or official IMD statements. If no warning is present in the context, explicitly confirm that there are no active warnings.
3. Do not override or contradict the risk engine scores, risk levels, or impact advisory recommendations.
4. Do not override the GRU model's predictions. When discussing next-hour or ML forecasts, reference the WeatherGRU model prediction as provided.
5. If the user asks about an activity, purpose, or decision (e.g. spraying, crop drying, harvesting, irrigation, travel, umbrella, outdoor events, construction, heat exposure):
   - Directly answer their question (e.g., whether conditions are favorable, marginal, or unfavorable, or whether an umbrella is needed) using the provided activity advisory analysis.
   - Ground your answer in the specific numerical meteorological parameters and thresholds provided.
   - Clearly distinguish between:
     a) Measured real-time observations
     b) Weather forecasts
     c) WeatherGPT+ operational advisories
     d) Official government / IMD warnings
   - Never make unconditional safety claims on weather alone; reference the specific meteorological criteria.
6. If any information or data source is unavailable (e.g. historical data, RAG search, or GRU prediction), explicitly state that it is unavailable rather than guessing.
7. Provide clear, concise, and actionable explanations suitable for general citizens, farmers, and travelers.
8. Maintain the requested output language strictly:
   - When language is 'en': Output in clear, fluent English.
   - When language is 'te': Output in natural, idiomatic Telugu (తెలుగు) rather than awkward word-for-word translation. Keep numbers, units (°C, km/h, mm, %), and location names accurate and clear.
9. The current user message must be classified independently before selecting tools. Conversation history may provide context such as location or date, but MUST NOT cause the previous intent or previous answer to be repeated. Greetings and acknowledgements are conversational intents and must not invoke weather tools.
"""

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, None
    
    model_name = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, model_name
    except Exception as e:
        logger.warning(f"Failed to initialize google-genai client: {e}")
        return None, model_name

def generate_grounded_response(
    user_message: str,
    context: Dict[str, Any],
    language: str = "en",
    conversation_history: Optional[list] = None
) -> Optional[str]:
    """
    Generates a natural language response grounded in backend context using Gemini.
    Returns None if Gemini is unavailable or fails, enabling deterministic fallback.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not configured. Falling back to deterministic response.")
        return None

    model_name = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    
    lang_instruction = (
        "Respond in natural, conversational Telugu (తెలుగు). Ensure weather numbers and units remain exact."
        if language == "te" else
        "Respond in clear, professional English."
    )

    history_section = ""
    if conversation_history:
        recent = conversation_history[-4:]
        lines = []
        for turn in recent:
            role = "User" if turn.get("role") == "user" else "Assistant"
            text = (turn.get("content") or turn.get("message") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        if lines:
            history_section = "\nRECENT CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n"

    context_json = json.dumps(context, indent=2, ensure_ascii=False)
    
    prompt = f"""USER QUESTION:
{user_message}

LANGUAGE REQUESTED:
{language} ({lang_instruction})
{history_section}
TRUSTED BACKEND CONTEXT:
```json
{context_json}
```

Please provide a grounded, helpful, and actionable response answering the user's question using ONLY the trusted context above."""


    # 1. Try official google-genai SDK
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GROUNDED_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1024,
            )
        )
        if response and response.text:
            return response.text.strip()
    except Exception as sdk_err:
        logger.warning(f"Gemini SDK generation failed: {sdk_err}. Trying REST API fallback.")

    # 2. Resilient REST API fallback via requests
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{GROUNDED_SYSTEM_PROMPT}\n\n{prompt}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024
            }
        }
        resp = requests.post(url, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
        else:
            logger.warning(f"Gemini REST API error {resp.status_code}: {resp.text}")
    except Exception as rest_err:
        logger.error(f"Gemini REST API call failed: {rest_err}")

    return None
