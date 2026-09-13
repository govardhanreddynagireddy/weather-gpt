from backend.agent.tool_router import route_tool

from backend.tools.gru_tool import (
    get_gru_info,
    predict_test_sample
)


# =====================================================
# ORCHESTRATOR
# =====================================================

def orchestrate(message):

    if not message or not message.strip():

        return {
            "success":False,
            "error":"Message cannot be empty."
        }


    # -------------------------------------------------
    # Determine tool
    # -------------------------------------------------

    tool=route_tool(message)


    # =================================================
    # GRU
    # =================================================

    if tool=="gru":

        prediction=predict_test_sample(
            index=0
        )


        return {

            "success":True,

            "tool":"gru",

            "message":message,

            "prediction":prediction,

            "model":get_gru_info()

        }


    # =================================================
    # RAG
    # =================================================

    if tool=="rag":

        return {

            "success":True,

            "tool":"rag",

            "message":message

        }


    # =================================================
    # HISTORICAL
    # =================================================

    if tool=="historical":

        return {

            "success":True,

            "tool":"historical",

            "message":message

        }


    # =================================================
    # RISK
    # =================================================

    if tool=="risk":

        return {

            "success":True,

            "tool":"risk",

            "message":message

        }


    # =================================================
    # GENERAL WEATHER
    # =================================================

    return {

        "success":True,

        "tool":"weather",

        "message":message

    }