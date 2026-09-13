from tools.weather_api_tool import get_weather
from tools.historical_tool import compare_weather
from services.risk_engine import calculate_risk
from services.impact_advisory import generate_advisory


weather=get_weather("Kurnool")

print("\nCURRENT WEATHER")
print(weather)


historical=compare_weather("Kurnool",weather)

print("\nHISTORICAL COMPARISON")
print(historical)


risk=calculate_risk(weather,historical)

print("\nRISK RESULT")
print(risk)


advisory=generate_advisory(weather,risk)

print("\nADVISORY")
print(advisory)