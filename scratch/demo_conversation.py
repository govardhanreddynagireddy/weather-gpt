import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("="*60)
print("DEMONSTRATION OF CONVERSATION FIX")
print("="*60)

# Step 1: Initial conversation turn establishing Ālampur
res0 = client.post("/chat", json={
    "message": "What is the weather in Ālampur?",
    "language": "en"
})
data0 = res0.json()
print("\n[User]: What is the weather in Ālampur?")
print(f"[Assistant]: {data0.get('answer')[:120]}...")
print(f"Tool used: {data0.get('tool')}, Location: {data0.get('location_context')}")

history = [
    {"role": "user", "content": "What is the weather in Ālampur?"},
    {"role": "assistant", "content": data0.get("answer", "")}
]

# Step 2: "What about tomorrow?"
res1 = client.post("/chat", json={
    "message": "What about tomorrow?",
    "language": "en",
    "conversation_history": history
})
data1 = res1.json()
print("\n[User]: What about tomorrow?")
print(f"[Assistant]:\n{data1.get('answer')}")
print(f"Tool used: {data1.get('tool')}, Type: {data1.get('type')}, Location: {data1.get('location_context')}")

history.extend([
    {"role": "user", "content": "What about tomorrow?"},
    {"role": "assistant", "content": data1.get("answer", "")}
])

# Step 3: "hi" -> MUST be greeting, MUST NOT call weather tool, MUST NOT repeat forecast!
print("\n" + "-"*40)
print("SENDING 'hi' MESSAGE:")
print("-"*40)
res2 = client.post("/chat", json={
    "message": "hi",
    "language": "en",
    "conversation_history": history
})
data2 = res2.json()
print("\n[User]: hi")
print(f"[Assistant]:\n{data2.get('answer')}")
print(f"Tool used: {data2.get('tool')}, Intent: {data2.get('intent')}, Location: {data2.get('location_context')}")

history.extend([
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": data2.get("answer", "")}
])

# Step 4: "will it rain?" -> Inherits Ālampur from context, answers rain information!
print("\n" + "-"*40)
print("SENDING 'will it rain?' MESSAGE:")
print("-"*40)
res3 = client.post("/chat", json={
    "message": "will it rain?",
    "language": "en",
    "conversation_history": history
})
data3 = res3.json()
print("\n[User]: will it rain?")
print(f"[Assistant]:\n{data3.get('answer')}")
print(f"Tool used: {data3.get('tool')}, Intent: {data3.get('intent')}, Location: {data3.get('location_context')}")
