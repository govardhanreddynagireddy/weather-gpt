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
        self.assertEqual(risk["risk_level"], "HIGH")

        advisory = generate_advisory(mock_weather, risk)
        self.assertEqual(advisory["risk_level"], "HIGH")
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


if __name__ == "__main__":
    unittest.main()
