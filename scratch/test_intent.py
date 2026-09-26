from backend.agent.intent_classifier import classify_user_intent, is_conversational_intent, get_conversational_response

queries = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "What about tomorrow?",
    "What is the weather in Alampur?",
    "will it rain?",
    "and rain?",
    "rain?",
    "is it safe for farming?",
    "Can I spray pesticides tomorrow morning?",
    "What about Hyderabad?",
    "show rain map"
]

for q in queries:
    intent = classify_user_intent(q)
    is_conv = is_conversational_intent(intent)
    print(f"Query: {q:40} -> Intent: {intent:20} -> Conversational: {is_conv}")
