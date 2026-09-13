# backend/tools/historical_tool.py

HISTORICAL_DATA={
    "Kurnool":{
        "temperature":32.0,
        "humidity":55.0,
        "rainfall":8.0,
        "wind_speed":3.0
    }
}


def compare_weather(city,current_weather):

    historical=HISTORICAL_DATA.get(city)

    if not historical:
        return {
            "available":False,
            "message":"Historical data is not available for this location."
        }

    result={}

    for parameter in ["temperature","humidity","rainfall","wind_speed"]:

        current=current_weather.get(parameter,0)
        average=historical[parameter]

        difference=current-average

        if average!=0:
            anomaly_percent=(difference/average)*100
        else:
            anomaly_percent=0

        result[parameter]={
            "current":current,
            "historical_average":average,
            "difference":round(difference,2),
            "anomaly_percent":round(anomaly_percent,2)
        }

    return {
        "available":True,
        "location":city,
        "comparison":result
    }