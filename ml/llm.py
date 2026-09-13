import requests


OLLAMA_URL="http://127.0.0.1:11434/api/generate"
MODEL="llama3.2:3b"


def ask_llm(prompt):

    response=requests.post(
        OLLAMA_URL,
        json={
            "model":MODEL,
            "prompt":prompt,
            "stream":False
        }
    )

    response.raise_for_status()

    return response.json()["response"]