"""
CORS Verification Script for WeatherGPT+
Tests preflight OPTIONS and actual requests with Vercel and local origins.
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

VERCEL_ORIGIN = "https://weather-gpt-eight-wine.vercel.app"
LOCAL_5500_ORIGIN = "http://127.0.0.1:5500"
LOCALHOST_5500_ORIGIN = "http://localhost:5500"

def test_endpoint_cors(name, method, path, body=None):
    print(f"\n==========================================")
    print(f"Testing {name}: {method} {path}")
    print(f"==========================================")
    
    # 1. Test OPTIONS Preflight
    print(f"1. OPTIONS Preflight for Vercel origin...")
    options_res = client.options(
        path,
        headers={
            "Origin": VERCEL_ORIGIN,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type"
        }
    )
    print(f"   Status Code: {options_res.status_code}")
    print(f"   Access-Control-Allow-Origin: {options_res.headers.get('access-control-allow-origin')}")
    print(f"   Access-Control-Allow-Credentials: {options_res.headers.get('access-control-allow-credentials')}")
    print(f"   Access-Control-Allow-Methods: {options_res.headers.get('access-control-allow-methods')}")
    assert options_res.headers.get("access-control-allow-origin") == VERCEL_ORIGIN, "OPTIONS failed allow-origin"
    assert options_res.headers.get("access-control-allow-credentials") == "true", "OPTIONS failed credentials"
    print("   [PASS] OPTIONS Preflight verified.")

    # 2. Test Actual Request with Vercel origin
    print(f"2. Actual {method} request with Vercel origin...")
    if method == "GET":
        res = client.get(path, headers={"Origin": VERCEL_ORIGIN})
    else:
        res = client.post(path, json=body or {}, headers={"Origin": VERCEL_ORIGIN})
    
    print(f"   Status Code: {res.status_code}")
    print(f"   Access-Control-Allow-Origin: {res.headers.get('access-control-allow-origin')}")
    print(f"   Access-Control-Allow-Credentials: {res.headers.get('access-control-allow-credentials')}")
    assert res.headers.get("access-control-allow-origin") == VERCEL_ORIGIN, "Request failed allow-origin"
    assert res.headers.get("access-control-allow-credentials") == "true", "Request failed credentials"
    print(f"   [PASS] Actual {method} request CORS verified.")

    # 3. Test Local Dev Origins (5500)
    for loc_origin in [LOCAL_5500_ORIGIN, LOCALHOST_5500_ORIGIN]:
        if method == "GET":
            l_res = client.get(path, headers={"Origin": loc_origin})
        else:
            l_res = client.post(path, json=body or {}, headers={"Origin": loc_origin})
        assert l_res.headers.get("access-control-allow-origin") == loc_origin, f"Failed for {loc_origin}"
    print(f"   [PASS] Local dev origins (5500) verified.")

if __name__ == "__main__":
    # Test /health
    test_endpoint_cors("Health Endpoint", "GET", "/health")

    # Test /api/weather
    test_endpoint_cors("Weather API", "GET", "/api/weather?lat=15.8281&lon=78.0373")

    # Test /chat
    test_endpoint_cors("Chat API", "POST", "/chat", body={"message": "hi", "language": "en"})

    print("\nALL CORS VERIFICATION TESTS PASSED SUCCESSFULLY!")
