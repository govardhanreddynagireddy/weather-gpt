import requests


# =====================================================
# OLLAMA SETTINGS
# =====================================================

OLLAMA_URL="http://127.0.0.1:11434/api/generate"

MODEL_NAME="llama3.2:3b"


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


    payload={
        "model":model,
        "prompt":prompt,
        "stream":False
    }


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


# =====================================================
# OLLAMA HEALTH CHECK
# =====================================================

def ollama_available():

    try:

        response=requests.get(
            "http://127.0.0.1:11434/api/tags",
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