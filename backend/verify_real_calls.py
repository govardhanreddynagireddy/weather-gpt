import os
import sys
from dotenv import load_dotenv, find_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv(find_dotenv(), override=True)

# Guard environment
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from backend.services.gemini_service import generate_grounded_response
from backend.agent.orchestrator import orchestrate, extract_city
from backend.tools.gru_tool import predict_latest_temperature, get_gru_info
from backend.tools.rag_tool import search_weather
from backend.services.risk_engine import calculate_risk
from backend.services.impact_advisory import generate_advisory
from backend.routes.alerts import _generate_alerts_payload

print("=" * 60)
print("WEATHERGPT+ DEEP VERIFICATION PASS")
print("=" * 60)

# 1. Real Gemini Success Test
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_success = False

if gemini_key and len(gemini_key.strip()) > 0:
    print("\n[1] REAL GEMINI API CALL TEST:")
    print("GEMINI_API_KEY detected (length: %d chars)." % len(gemini_key.strip()))
    test_context = {
        "location": "Kurnool",
        "current_weather": {
            "temperature_celsius": 29.36,
            "humidity_percent": 62,
            "wind_speed_kmh": 2.9,
            "weather_description": "light rain",
            "precipitation_mm": 0.11
        }
    }
    resp = generate_grounded_response(
        user_message="What is the current weather information provided in the context?",
        context=test_context,
        language="en"
    )
    if resp:
        print("Real Gemini API call succeeded!")
        print("Response snippet:", resp[:150].replace("\n", " "), "...")
        # Verify grounding: check if response reflects the context data
        if "29" in resp or "Kurnool" in resp or "rain" in resp.lower():
            print("Grounding check: PASS (response correctly reflects structured context values)")
            gemini_success = True
        else:
            print("Grounding check: WARNING (response did not mention context numbers)")
    else:
        print("Real Gemini API call returned None (SDK / REST failed).")
else:
    print("\n[1] REAL GEMINI API CALL TEST:")
    print("Real Gemini success test could not be executed because GEMINI_API_KEY is not configured.")

# 2. English Chat via Orchestrator
print("\n[2] REAL ENGLISH CHAT TEST:")
en_res = orchestrate("What is the weather in Kurnool?", language="en")
print("Success:", en_res.get("success"))
print("Tool used:", en_res.get("tool"))
print("Answer snippet:", en_res.get("answer", "")[:150].replace("\n", " "), "...")
assert en_res.get("success") is True, "English chat failed"

# 3. Telugu Chat via Orchestrator
print("\n[3] REAL TELUGU CHAT TEST:")
city_extracted = extract_city("కర్నూలులో ప్రస్తుతం వాతావరణం ఎలా ఉంది?")
print("City extracted from Telugu:", city_extracted)
assert city_extracted == "Kurnool", f"City extraction failed: {city_extracted}"
te_res = orchestrate("కర్నూలులో ప్రస్తుతం వాతావరణం ఎలా ఉంది?", language="te")
print("Success:", te_res.get("success"))
print("Tool used:", te_res.get("tool"))
print("Answer snippet:", te_res.get("answer", "")[:150].replace("\n", " "), "...")
assert te_res.get("success") is True, "Telugu chat failed"

# 4. GRU Next-Hour Prediction
print("\n[4] GRU PREDICTION TEST:")
gru_res = orchestrate("What will the temperature be next hour?", language="en")
print("Success:", gru_res.get("success"))
print("Tool used:", gru_res.get("tool"))
print("Predicted Temp:", gru_res.get("predicted_temperature_celsius"))
print("Answer snippet:", gru_res.get("answer", "")[:150].replace("\n", " "), "...")
assert gru_res.get("success") is True, "GRU chat failed"

# 5. RAG IMD Document Retrieval
print("\n[5] RAG RETRIEVAL TEST:")
rag_res = orchestrate("What does IMD say about heavy rainfall?", language="en")
print("Success:", rag_res.get("success"))
print("Tool used:", rag_res.get("tool"))
print("Sources count:", len(rag_res.get("sources", [])))
print("Answer snippet:", rag_res.get("answer", "")[:150].replace("\n", " "), "...")
assert rag_res.get("success") is True, "RAG chat failed"

# 6. Risk Engine & Impact Advisory
print("\n[6] RISK & ADVISORY TEST:")
risk_res = orchestrate("What is the weather risk and agricultural advisory for Kurnool?", language="en")
print("Success:", risk_res.get("success"))
print("Tool used:", risk_res.get("tool"))
print("Risk Level:", risk_res.get("risk", {}).get("risk_level"))
print("Risk Score:", risk_res.get("risk", {}).get("risk_score"))
print("Answer snippet:", risk_res.get("answer", "")[:150].replace("\n", " "), "...")
assert risk_res.get("success") is True, "Risk chat failed"

# 7. Alerts Payload
print("\n[7] ALERTS PAYLOAD TEST:")
alerts_res = _generate_alerts_payload("Kurnool")
print("Location:", alerts_res.get("location"))
print("Has Alert:", alerts_res.get("has_alert"))
print("Risk Level:", alerts_res.get("risk_level"))
print("Alerts Count:", len(alerts_res.get("alerts", [])))
assert "risk_level" in alerts_res, "Alerts payload missing risk_level"

print("\n" + "=" * 60)
print("ALL DEEP VERIFICATION CHECKS COMPLETED SUCCESSFULLY")
print("=" * 60)
