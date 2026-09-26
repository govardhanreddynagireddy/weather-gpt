"""
Comprehensive verification script for WeatherGPT+ Task 24 & Task 25 requirements.
Tests:
1. Local health endpoint & UTF-8 headers
2. All 12 required queries from Task 24
3. Typo queries (tommorow, doot)
4. Production Railway health & chat endpoints
"""

import json
import requests
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LOCAL_API = "http://127.0.0.1:8000"
PROD_API = "https://weather-gpt-production-4837.up.railway.app"

def test_local_health():
    time.sleep(2)
    print("--- 1. Testing Local Health & Headers ---")
    try:
        r = requests.get(f"{LOCAL_API}/health", timeout=5)
        print("Status:", r.status_code)
        print("Content-Type:", r.headers.get("content-type"))
        print("Body:", r.text)
        assert r.status_code == 200
        assert "application/json" in r.headers.get("content-type", "")
        print("Local health PASS\n")
    except Exception as e:
        print("Local health FAILED:", e, "\n")

def test_queries():
    print("--- 2. Testing Task 24 Required Queries Locally ---")
    queries = [
        ("1. hi", "hi", "en", None),
        ("2. hello", "hello", "en", None),
        ("3. Kurnool weather", "What is the weather in Kurnool?", "en", None),
        ("4. What about tomorrow?", "What about tomorrow?", "en", [{"role": "user", "content": "What is the weather in Kurnool?"}]),
        ("5. Can I go outside tomorrow?", "Can I go outside tomorrow?", "en", [{"role": "user", "content": "What about tomorrow in Kurnool?"}]),
        ("6. Can I spray pesticide tomorrow?", "Can I spray pesticide tomorrow?", "en", [{"role": "user", "content": "What about tomorrow in Kurnool?"}]),
        ("7. Will it rain tomorrow?", "Will it rain tomorrow?", "en", [{"role": "user", "content": "What about tomorrow in Kurnool?"}]),
        ("8. repu weather ela untadhi", "repu weather ela untadhi", "te", [{"role": "user", "content": "కర్నూలు"}]),
        ("9. రేపు వాతావరణం ఎలా ఉంటుంది?", "రేపు వాతావరణం ఎలా ఉంటుంది?", "te", None),
        ("10. What about tomorrow in Kurnool?", "What about tomorrow in Kurnool?", "en", None),
        ("11. What about the rain?", "What about the rain?", "en", [{"role": "user", "content": "What about tomorrow in Kurnool?"}]),
        ("12. Can I dry harvested crops tomorrow?", "Can I dry harvested crops tomorrow?", "en", [{"role": "user", "content": "What is the weather in Kurnool?"}]),
        ("13. Typo: tommorow weather", "tommorow weather", "en", None),
        ("14. Typo: can I go doot tomorrow?", "can I go doot tomorrow?", "en", None)
    ]

    for label, msg, lang, hist in queries:
        print(f"Testing [{label}] -> msg: '{msg}' (lang: {lang})")
        payload = {
            "message": msg,
            "language": lang,
            "location_context": {"city": "Kurnool"},
            "conversation_history": hist or []
        }
        try:
            r = requests.post(f"{LOCAL_API}/chat", json=payload, timeout=20)
            data = r.json()
            tool = data.get("tool")
            typ = data.get("type")
            city = (data.get("location_context") or {}).get("city")
            answer = (data.get("answer") or "")[:90].replace("\n", " ")
            pills_count = len(data.get("action_pills") or [])
            print(f"  Status: {r.status_code} | Tool: {tool} | Type: {typ} | City: {city} | Pills: {pills_count}")
            print(f"  Answer: {answer}...\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

def test_production():
    print("--- 3. Testing Railway Production API ---")
    try:
        r = requests.get(f"{PROD_API}/health", timeout=10)
        print("Railway /health Status:", r.status_code)
        print("Railway /health Body:", r.text)
    except Exception as e:
        print("Railway /health FAILED:", e)

    try:
        payload = {"message": "What about tomorrow?", "language": "en"}
        r = requests.post(f"{PROD_API}/chat", json=payload, timeout=25)
        print("Railway /chat Status:", r.status_code)
        data = r.json()
        print("Railway /chat success:", data.get("success"))
        print("Railway /chat tool:", data.get("tool"))
        print("Railway /chat answer:", (data.get("answer") or "")[:80])
    except Exception as e:
        print("Railway /chat FAILED:", e)

if __name__ == "__main__":
    test_local_health()
    test_queries()
    test_production()
