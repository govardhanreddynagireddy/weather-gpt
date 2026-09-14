import os
import requests


# =====================================================
# OLLAMA SETTINGS
# =====================================================

OLLAMA_URL=os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate"
)

MODEL_NAME=os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)


# =====================================================
# GENERATE RESPONSE
# =====================================================

def generate_response(
    prompt,
    model=MODEL_NAME
):

    if not prompt or not prompt.strip():

        raise ValueError(
            "Prompt cannot be empty."
        )


    if not ollama_available():

        return (
            "Ollama is currently unavailable. "
            "The AI response service is not running."
        )


    payload={
        "model":model,
        "prompt":prompt,
        "stream":False
    }


    try:

        response=requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data=response.json()

        return data.get(
            "response",
            ""
        ).strip()

    except requests.RequestException:

        return (
            "Ollama is currently unavailable. "
            "Please try again later."
        )


# =====================================================
# OLLAMA HEALTH CHECK
# =====================================================

def ollama_available():

    try:

        tags_url=OLLAMA_URL.replace(
            "/api/generate",
            "/api/tags"
        )

        response=requests.get(
            tags_url,
            timeout=5
        )

        return response.status_code==200

    except requests.RequestException:

        return False


# =====================================================
# MODEL INFO
# =====================================================

def get_ollama_info():

    return {

        "provider":"Ollama",

        "model":MODEL_NAME,

        "url":OLLAMA_URL,

        "available":ollama_available()

    }