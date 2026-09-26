import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Guard environment
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi.testclient import TestClient
from backend.main import app
from backend.tools.weather_tool import get_weather, get_weather_forecast
from backend.tools.historical_tool import compare_weather
from backend.services.risk_engine import calculate_risk
from backend.services.impact_advisory import generate_advisory
from backend.tools.gru_tool import predict_latest_temperature, get_gru_info
from backend.tools.rag_tool import search_weather
from backend.services.gemini_service import generate_grounded_response
from backend.agent.orchestrator import orchestrate, extract_city


class TestWeatherGPTSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # A. Application startup & B. Health endpoint
    def test_01_root_and_health(self):
        root_res = self.client.get("/")
        self.assertEqual(root_res.status_code, 200)
        self.assertEqual(root_res.json().get("status"), "success")

        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertEqual(health_res.json().get("status"), "healthy")

    # C. Current weather & D. Forecast
    def test_02_weather_and_forecast(self):
        weather_res = self.client.get("/weather/Kurnool")
        self.assertIn(weather_res.status_code, [200, 502])
        if weather_res.status_code == 200:
            data = weather_res.json()
            self.assertIn("temperature_celsius", data)
            self.assertIn("humidity_percent", data)
            self.assertIn("wind_speed_kmh", data)

    # R. Invalid location & S. Weather API failure handling
    def test_03_invalid_location_handling(self):
        res = self.client.get("/weather/undefined")
        self.assertEqual(res.status_code, 502)
        data = res.json()
        self.assertFalse(data.get("success", True))

    # L. Risk engine & M. Impact advisory
    def test_04_risk_engine_and_advisory(self):
        mock_weather = {
            "temperature_celsius": 42.0,
            "humidity_percent": 95,
            "wind_speed_kmh": 22.0,
            "precipitation_mm": 60.0
        }
        mock_hist = {
            "available": True,
            "comparison": {
                "temperature": {"anomaly_percent": 25.0},
                "wind_speed": {"anomaly_percent": 35.0}
            }
        }
        risk = calculate_risk(mock_weather, mock_hist)
        self.assertIn("risk_score", risk)
        self.assertIn("risk_level", risk)
        self.assertIn(risk["risk_level"], ["HIGH", "SEVERE"])

        advisory = generate_advisory(mock_weather, risk)
        self.assertIn(advisory["risk_level"], ["HIGH", "SEVERE"])
        self.assertTrue(len(advisory["possible_impacts"]) > 0)
        self.assertTrue(len(advisory["recommendations"]) > 0)

    # N. Alerts endpoint
    def test_05_alerts_endpoint(self):
        res = self.client.get("/alerts/Kurnool")
        self.assertIn(res.status_code, [200, 502])
        if res.status_code == 200:
            data = res.json()
            self.assertIn("has_alert", data)
            self.assertIn("risk_level", data)
            self.assertIn("alerts", data)

    # Feedback endpoint
    def test_06_feedback_endpoint(self):
        res = self.client.post("/feedback", json={"rating": 5, "comment": "Great weather assistant!"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json().get("success"))

    # I. GRU prediction
    def test_07_gru_prediction(self):
        try:
            pred = predict_latest_temperature()
            self.assertTrue(isinstance(pred, (float, int, dict)))
            info = get_gru_info()
            self.assertIn("model", info)
        except Exception as e:
            self.fail(f"GRU prediction failed with exception: {e}")

    # J. RAG retrieval & K. IMD-grounded response
    def test_08_rag_retrieval(self):
        try:
            results = search_weather("IMD rainfall warning forecast")
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"RAG search failed with exception: {e}")

    # F. Gemini integration & O. English chat & P. Telugu chat
    def test_09_chat_english_and_telugu(self):
        # English chat
        en_res = self.client.post("/chat", json={"message": "What is the weather in Kurnool?", "language": "en"})
        self.assertEqual(en_res.status_code, 200)
        en_data = en_res.json()
        self.assertTrue(en_data.get("success"))
        self.assertIn("answer", en_data)

        # Telugu chat
        te_res = self.client.post("/chat", json={"message": "కర్నూలులో వాతావరణం ఎలా ఉంది?", "language": "te"})
        self.assertEqual(te_res.status_code, 200)
        te_data = te_res.json()
        self.assertTrue(te_data.get("success"))
        self.assertIn("answer", te_data)

    # G. Missing Gemini key fallback
    def test_10_gemini_missing_key_fallback(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            res = self.client.post("/chat", json={"message": "What is the next hour temperature?", "language": "en"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data.get("success"))
            self.assertIn("Next-hour temperature prediction", data.get("answer", ""))

    # H. Gemini API failure fallback
    def test_11_gemini_failure_fallback(self):
        with patch("backend.services.gemini_service.generate_grounded_response", return_value=None):
            res = self.client.post("/chat", json={"message": "What is the weather in Kurnool?", "language": "en"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data.get("success"))
            self.assertTrue(len(data.get("answer", "")) > 0)

    # Q. Multi-turn Conversational Context Retention
    def test_12_conversational_history_followup(self):
        # Turn 1: Initial query with explicit city
        t1_res = self.client.post("/chat", json={
            "message": "What is the weather in Kurnool?",
            "language": "en"
        })
        self.assertEqual(t1_res.status_code, 200)
        t1_data = t1_res.json()
        self.assertTrue(t1_data.get("success"))

        history = [
            {"role": "user", "content": "What is the weather in Kurnool?"},
            {"role": "assistant", "content": t1_data.get("answer", "")}
        ]

        # Turn 2: Follow-up query asking about tomorrow (no city specified)
        t2_res = self.client.post("/chat", json={
            "message": "What about tomorrow?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(t2_res.status_code, 200)
        t2_data = t2_res.json()
        self.assertTrue(t2_data.get("success"))
        self.assertEqual(t2_data.get("location_context", {}).get("city"), "Kurnool")
        self.assertEqual(t2_data.get("location_context", {}).get("days_ahead"), 1)
        self.assertEqual(t2_data.get("type"), "tomorrow_forecast")

        history.extend([
            {"role": "user", "content": "What about tomorrow?"},
            {"role": "assistant", "content": t2_data.get("answer", "")}
        ])

        # Turn 3: Short follow-up "Rain?" inheriting tomorrow & Kurnool
        t3_res = self.client.post("/chat", json={
            "message": "Rain?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(t3_res.status_code, 200)
        t3_data = t3_res.json()
        self.assertTrue(t3_data.get("success"))
        self.assertEqual(t3_data.get("location_context", {}).get("city"), "Kurnool")
        self.assertEqual(t3_data.get("location_context", {}).get("days_ahead"), 1)

        history.extend([
            {"role": "user", "content": "Rain?"},
            {"role": "assistant", "content": t3_data.get("answer", "")}
        ])

        # Turn 4: Farming advisory follow-up
        t4_res = self.client.post("/chat", json={
            "message": "Is it safe for farming?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(t4_res.status_code, 200)
        t4_data = t4_res.json()
        self.assertTrue(t4_data.get("success"))
        self.assertEqual(t4_data.get("tool"), "risk")
        self.assertEqual(t4_data.get("location_context", {}).get("city"), "Kurnool")

    # 13. General-Purpose Conversational Weather: Pesticide Spraying
    def test_13_activity_pesticide_spraying(self):
        res = self.client.post("/chat", json={
            "message": "Can I spray pesticides tomorrow morning in Kurnool?",
            "language": "en"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("activity"), "pesticide_spraying")
        self.assertEqual(data.get("domain"), "agriculture")
        self.assertIn("activity_advisory", data)
        advisory = data["activity_advisory"]
        self.assertIn(advisory.get("suitability"), ["FAVORABLE", "MARGINAL", "UNFAVORABLE"])
        self.assertIn("attributions", advisory)
        self.assertIn("observation", advisory["attributions"])
        self.assertIn("forecast", advisory["attributions"])
        self.assertIn("official_warnings", advisory["attributions"])
        self.assertTrue(len(data.get("answer", "")) > 0)

    # 14. General-Purpose Conversational Weather: Crop Drying
    def test_14_activity_crop_drying(self):
        res = self.client.post("/chat", json={
            "message": "Can I dry my harvested crop tomorrow in Kurnool?",
            "language": "en"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("activity"), "crop_drying")
        self.assertEqual(data.get("domain"), "agriculture")
        self.assertIn("activity_advisory", data)
        self.assertTrue(len(data.get("answer", "")) > 0)

    # 15. General-Purpose Conversational Weather: Travel & Umbrella
    def test_15_activity_umbrella_rain_gear(self):
        res = self.client.post("/chat", json={
            "message": "Should I carry an umbrella when I travel to Kurnool tomorrow?",
            "language": "en"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("activity"), "umbrella_rain_gear")
        self.assertEqual(data.get("domain"), "travel_commuting")
        self.assertIn("activity_advisory", data)
        self.assertTrue(len(data.get("answer", "")) > 0)

    # 16. General-Purpose Conversational Weather: Telugu Natural Language Activity Query
    def test_16_activity_telugu_query(self):
        res = self.client.post("/chat", json={
            "message": "రేపు కర్నూలులో పంట ఆరబెట్టవచ్చా?",
            "language": "te"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("activity"), "crop_drying")
        self.assertIn("activity_advisory", data)
        self.assertTrue(len(data.get("answer", "")) > 0)

    # ---------------------------------------------------------
    # CONVERSATIONAL AGENT INTENT & MEMORY VERIFICATION (TEST 1 - 6)
    # ---------------------------------------------------------

    # TEST 1: "What about tomorrow?" then "hi" -> Second response is greeting, NOT forecast.
    def test_17_intent_sequence_test1(self):
        # Step 1: "What about tomorrow?"
        res1 = self.client.post("/chat", json={
            "message": "What about tomorrow?",
            "language": "en"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertTrue(data1.get("success"))
        self.assertEqual(data1.get("type"), "tomorrow_forecast")

        history = [
            {"role": "user", "content": "What about tomorrow?"},
            {"role": "assistant", "content": data1.get("answer", "")}
        ]

        # Step 2: "hi" -> Must be greeting, must NOT repeat forecast or call weather API
        res2 = self.client.post("/chat", json={
            "message": "hi",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2.get("tool"), "conversational")
        self.assertEqual(data2.get("intent"), "greeting")
        self.assertIn("Hello", data2.get("answer", ""))
        self.assertNotIn("Tomorrow's forecast", data2.get("answer", ""))

    # TEST 2: "What is the weather in Ālampur?" then "will it rain?" -> Rain info for Ālampur.
    def test_18_intent_sequence_test2(self):
        res1 = self.client.post("/chat", json={
            "message": "What is the weather in Ālampur?",
            "language": "en"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertTrue(data1.get("success"))
        self.assertEqual(data1.get("location_context", {}).get("city"), "Alampur")

        history = [
            {"role": "user", "content": "What is the weather in Ālampur?"},
            {"role": "assistant", "content": data1.get("answer", "")}
        ]

        # Step 2: "will it rain?" -> Inherits Alampur, returns rain info
        res2 = self.client.post("/chat", json={
            "message": "will it rain?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2.get("location_context", {}).get("city"), "Alampur")
        self.assertIn("rain", data2.get("answer", "").lower())

    # TEST 3: "What is the weather in Ālampur?" then "hi" then "will it rain?"
    def test_19_intent_sequence_test3(self):
        # 1. Weather
        res1 = self.client.post("/chat", json={
            "message": "What is the weather in Ālampur?",
            "language": "en"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1.get("location_context", {}).get("city"), "Alampur")

        history = [
            {"role": "user", "content": "What is the weather in Ālampur?"},
            {"role": "assistant", "content": data1.get("answer", "")}
        ]

        # 2. Greeting
        res2 = self.client.post("/chat", json={
            "message": "hi",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2.get("tool"), "conversational")
        self.assertEqual(data2.get("intent"), "greeting")
        self.assertIn("Hello", data2.get("answer", ""))
        self.assertEqual(data2.get("location_context", {}).get("city"), "Alampur")

        history.extend([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": data2.get("answer", "")}
        ])

        # 3. Rain query for Alampur
        res3 = self.client.post("/chat", json={
            "message": "will it rain?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res3.status_code, 200)
        data3 = res3.json()
        self.assertTrue(data3.get("success"))
        self.assertEqual(data3.get("location_context", {}).get("city"), "Alampur")
        self.assertIn("rain", data3.get("answer", "").lower())

    # TEST 4: "What about tomorrow?" then "thanks" -> Short acknowledgement.
    def test_20_intent_sequence_test4(self):
        res1 = self.client.post("/chat", json={
            "message": "What about tomorrow?",
            "language": "en"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()

        history = [
            {"role": "user", "content": "What about tomorrow?"},
            {"role": "assistant", "content": data1.get("answer", "")}
        ]

        res2 = self.client.post("/chat", json={
            "message": "thanks",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2.get("tool"), "conversational")
        self.assertEqual(data2.get("intent"), "thanks")
        self.assertIn("welcome", data2.get("answer", "").lower())

    # TEST 5: "What is the weather in Ālampur?" then "What about tomorrow?" -> Tomorrow forecast for Ālampur.
    def test_21_intent_sequence_test5(self):
        res1 = self.client.post("/chat", json={
            "message": "What is the weather in Ālampur?",
            "language": "en"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1.get("location_context", {}).get("city"), "Alampur")

        history = [
            {"role": "user", "content": "What is the weather in Ālampur?"},
            {"role": "assistant", "content": data1.get("answer", "")}
        ]

        res2 = self.client.post("/chat", json={
            "message": "What about tomorrow?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2.get("type"), "tomorrow_forecast")
        self.assertEqual(data2.get("location_context", {}).get("city"), "Alampur")

    # TEST 6: "What is the weather in Ālampur?" then "What about Hyderabad?" -> Hyderabad weather, not Ālampur.
    def test_22_intent_sequence_test6(self):
        res1 = self.client.post("/chat", json={
            "message": "What is the weather in Ālampur?",
            "language": "en"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1.get("location_context", {}).get("city"), "Alampur")

        history = [
            {"role": "user", "content": "What is the weather in Ālampur?"},
            {"role": "assistant", "content": data1.get("answer", "")}
        ]

        res2 = self.client.post("/chat", json={
            "message": "What about Hyderabad?",
            "language": "en",
            "conversation_history": history
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2.get("location_context", {}).get("city"), "Hyderabad")
        self.assertEqual(data2.get("type"), "city_weather")


if __name__ == "__main__":
    unittest.main()

