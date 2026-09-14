from backend.agent.tool_router import route_tool

from backend.tools.gru_tool import (
    get_gru_info,
    predict_latest_temperature
)

from backend.services.ollama_service import (
    generate_response
)

from backend.tools.rag_tool import (
    search_weather
)


# =====================================================
# ORCHESTRATOR
# =====================================================

def orchestrate(message):

    # =================================================
    # VALIDATE MESSAGE
    # =================================================

    if not message or not message.strip():

        return {
            "success":False,
            "error":"Message cannot be empty."
        }


    message=message.strip()

    tool=route_tool(message)


    # =================================================
    # GENERAL WEATHER → OLLAMA
    # =================================================

    if tool=="weather":

        prompt=f"""
You are WeatherGPT+, a helpful weather assistant.

User question:
{message}

Answer the user's question clearly and concisely.

If the question requires live weather data that is
not provided to you, do not invent weather information.

Instead, clearly say that live weather data is required.
"""

        answer=generate_response(
            prompt
        )

        return {
            "success":True,
            "tool":"weather",
            "message":message,
            "answer":answer
        }


    # =================================================
    # GRU → NEXT TEMPERATURE
    # =================================================

    if tool=="gru":

        prediction=predict_latest_temperature()

        return {
            "success":True,
            "tool":"gru",
            "message":message,
            "prediction":prediction,
            "model":get_gru_info(),
            "source":"era5_merged.nc"
        }


    # =================================================
    # RAG → FAISS → OLLAMA
    # =================================================

    if tool=="rag":

        results=search_weather(
            message
        )


        # -------------------------------------------------
        # No relevant documents
        # -------------------------------------------------

        if not results:

            return {
                "success":True,
                "tool":"rag",
                "message":message,
                "answer":(
                    "No relevant information was found "
                    "in the IMD documents."
                ),
                "sources":[]
            }


        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context=""

        for i,result in enumerate(
            results,
            1
        ):

            context+=f"""
SOURCE {i}

DOCUMENT:
{result["source"]}

CHUNK:
{result["chunk_id"]}

RELEVANCE:
{result["score"]}

TEXT:
{result["text"]}

"""


        # -------------------------------------------------
        # RAG → OLLAMA
        # -------------------------------------------------

        prompt=f"""
You are WeatherGPT+, an official weather information
assistant.

Your task is to answer the user's question using ONLY
the IMD weather context provided below.

IMPORTANT RULES:

1. Carefully read ALL the IMD context before answering.

2. If the answer is present anywhere in the context,
   you MUST answer the user's question.

3. Do NOT say that the information is unavailable if
   the information is explicitly present in the context.

4. Do NOT use outside knowledge.

5. Do NOT invent weather information.

6. If multiple documents contain similar information,
   prefer the latest IMD document.

7. Pay attention to the actual forecast date mentioned
   in the document.

8. If the user asks about "tomorrow", determine the
   correct forecast date from the IMD context.

9. Mention the rainfall intensity and affected locations
   when they are available.

10. If warning information is present, explain it
    directly instead of refusing to answer.

11. Keep the answer concise and easy to understand.

USER QUESTION:
{message}

IMD WEATHER CONTEXT:
{context}

Now answer the user's question directly.
"""


        answer=generate_response(
            prompt
        )


        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources=[]

        for result in results:

            sources.append({

                "source":result["source"],

                "chunk_id":result["chunk_id"],

                "score":result["score"]

            })


        return {

            "success":True,

            "tool":"rag",

            "message":message,

            "answer":answer,

            "sources":sources

        }


    # =================================================
    # HISTORICAL
    # =================================================

    if tool=="historical":

        return {

            "success":True,

            "tool":"historical",

            "message":message,

            "answer":(
                "Historical weather analysis "
                "is not connected yet."
            )

        }


    # =================================================
    # RISK
    # =================================================

    if tool=="risk":

        return {

            "success":True,

            "tool":"risk",

            "message":message,

            "answer":(
                "Weather risk analysis "
                "is not connected yet."
            )

        }


    # =================================================
    # FALLBACK → OLLAMA
    # =================================================

    prompt=f"""
You are WeatherGPT+.

User question:
{message}

Give a helpful and concise answer.

Do not invent live weather information.
"""

    answer=generate_response(
        prompt
    )

    return {

        "success":True,

        "tool":"weather",

        "message":message,

        "answer":answer

    }