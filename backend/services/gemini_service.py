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
5. If any information or data source is unavailable (e.g. historical data, RAG search, or GRU prediction), explicitly state that it is unavailable rather than guessing.
6. Provide clear, concise, and actionable explanations suitable for general citizens, farmers, and travelers.
7. Maintain the requested output language strictly:
   - When language is 'en': Output in clear, fluent English.
   - When language is 'te': Output in natural, idiomatic Telugu (తెలుగు) rather than awkward word-for-word translation. Keep numbers, units (°C, km/h, mm, %), and location names accurate and clear.
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
    language: str = "en"
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

    context_json = json.dumps(context, indent=2, ensure_ascii=False)
    
    prompt = f"""USER QUESTION:
{user_message}

LANGUAGE REQUESTED:
{language} ({lang_instruction})

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
