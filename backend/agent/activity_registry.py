"""
Activity & Purpose Registry for WeatherGPT+
Defines a generalized, extensible catalog of user purposes, activity domains,
required meteorological variables, and calibrated decision thresholds.
"""

import re
from typing import Dict, Any, List, Optional

ACTIVITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # 1. AGRICULTURE
    # -------------------------------------------------------------------------
    "pesticide_spraying": {
        "key": "pesticide_spraying",
        "domain": "agriculture",
        "name": "Pesticide & Chemical Spraying",
        "decision_context": "spray_drift_and_washoff",
        "aliases_en": [
            r"\bspray\b", r"\bspraying\b", r"\bpesticide\b", r"\binsecticide\b",
            r"\bfungicide\b", r"\bweedicide\b", r"\bherbicide\b", r"\bfoliar\b",
            r"\bchemical spray\b"
        ],
        "aliases_te": [
            "స్ప్రే", "మందు కొట్ట", "మందు పిచికారీ", "పురుగుల మందు",
            "క్రిమిసంహారక", "పిచికారీ", "mandu kotta", "spray"
        ],
        "required_variables": ["wind_speed", "precipitation", "temperature", "humidity"],
        "thresholds": {
            "max_wind_kmh": 15.0,        # Drift hazard above 15 km/h
            "max_rain_mm": 0.2,          # Rain wash-off risk
            "max_temp_c": 36.0           # Excessive chemical evaporation/leaf burn
        },
        "default_favorable_advice": "Wind and rain conditions are favorable for spraying. Apply in early morning or late afternoon.",
        "default_unfavorable_advice": "Postpone spraying due to wind drift or precipitation wash-off hazards."
    },
    "crop_drying": {
        "key": "crop_drying",
        "domain": "agriculture",
        "name": "Crop & Grain Drying",
        "decision_context": "sun_drying_and_spoilage",
        "aliases_en": [
            r"\bdry\b.*\bcrop\b", r"\bcrop\b.*\bdry\b", r"\bdry\b.*\bgrain\b",
            r"\bgrain\b.*\bdry\b", r"\bdrying\b", r"\bsun\b.*\bdry\b",
            r"\bchilli\b.*\bdry\b", r"\bpaddy\b.*\bdry\b",
            r"\bdry\b.*\bharvest", r"\bharvested\s+crop"
        ],
        "aliases_te": [
            "ఆరబెట్ట", "ఎండబెట్ట", "ధాన్యం ఆర", "పంట ఆర", "మిరప ఆర",
            "aarabett", "endabett", "drying"
        ],
        "required_variables": ["precipitation", "humidity", "clouds", "temperature"],
        "thresholds": {
            "max_rain_mm": 0.0,          # Zero rain tolerance for open-air drying
            "max_humidity_percent": 72.0, # High ambient humidity prevents rapid drying
            "min_temp_c": 22.0           # Requires warm solar insulation
        },
        "default_favorable_advice": "Conditions are favorable for open-air crop drying with ample sunshine and dry air.",
        "default_unfavorable_advice": "Do not leave crops spread in the open; rain or high ambient moisture will cause mold/spoilage."
    },
    "crop_harvesting": {
        "key": "crop_harvesting",
        "domain": "agriculture",
        "name": "Crop Harvesting",
        "decision_context": "harvest_moisture_and_lodging",
        "aliases_en": [
            r"\bharvest\b", r"\bharvesting\b", r"\bcutting\b.*\bcrop\b",
            r"\breaping\b", r"\bthreshing\b", r"\bcombine\b.*\bharvest\b"
        ],
        "aliases_te": [
            "కోత", "పంట కోత", "కోయవచ్చా", "నూర్పిడి", "kotha", "harvest"
        ],
        "required_variables": ["precipitation", "wind_speed", "soil_moisture"],
        "thresholds": {
            "max_rain_mm": 1.0,          # Machinery bogging down & wet grain rot
            "max_wind_kmh": 28.0         # Risk of crop lodging
        },
        "default_favorable_advice": "Dry conditions make it suitable for harvesting and mechanical field operations.",
        "default_unfavorable_advice": "Postpone harvest operations to prevent grain moisture damage and soil compaction."
    },
    "irrigation": {
        "key": "irrigation",
        "domain": "agriculture",
        "name": "Irrigation & Field Watering",
        "decision_context": "soil_water_balance",
        "aliases_en": [
            r"\birrigate\b", r"\birrigation\b", r"\bwater\b.*\bfield\b",
            r"\bwatering\b.*\bcrop\b", r"\bborewell\b"
        ],
        "aliases_te": [
            "నీరు పెట్ట", "నీటిపారుదల", "తడులు", "పొలానికి నీరు", "neeru petta", "irrigation"
        ],
        "required_variables": ["precipitation", "temperature", "humidity"],
        "thresholds": {
            "rain_saving_mm": 5.0        # Natural rain > 5mm saves an irrigation cycle
        },
        "default_favorable_advice": "Proceed with regular irrigation schedule as natural precipitation is insufficient.",
        "default_unfavorable_advice": "Suspend irrigation; expected natural rainfall will provide adequate soil moisture."
    },
    "fertilizing": {
        "key": "fertilizing",
        "domain": "agriculture",
        "name": "Fertilizer Application",
        "decision_context": "nutrient_leaching_and_runoff",
        "aliases_en": [
            r"\bfertilizer\b", r"\burea\b", r"\bdap\b", r"\bfertilizing\b",
            r"\bmanure\b", r"\bnutrient application\b"
        ],
        "aliases_te": [
            "ఎరువులు", "యూరియా", "ఎరువు వేయ", "eruvulu", "fertilizer"
        ],
        "required_variables": ["precipitation", "wind_speed"],
        "thresholds": {
            "max_rain_mm": 5.0           # Leaching / surface runoff risk
        },
        "default_favorable_advice": "Adequate soil moisture with no torrential downpour expected; good for fertilizer application.",
        "default_unfavorable_advice": "Avoid fertilizer broadcasting; expected runoff will wash nutrients away into drains."
    },

    # -------------------------------------------------------------------------
    # 2. TRAVEL, COMMUTING & RAIN GEAR
    # -------------------------------------------------------------------------
    "umbrella_rain_gear": {
        "key": "umbrella_rain_gear",
        "domain": "travel_commuting",
        "name": "Umbrella & Rain Protection",
        "decision_context": "precipitation_preparedness",
        "aliases_en": [
            r"\bumbrella\b", r"\braincoat\b", r"\brain gear\b",
            r"\bcarry an umbrella\b", r"\bneed an umbrella\b"
        ],
        "aliases_te": [
            "గొడుగు", "రెయిన్‌కోట్", "godugu", "umbrella"
        ],
        "required_variables": ["precipitation", "weather_condition"],
        "thresholds": {
            "rain_trigger_mm": 0.3       # Any measurable rain calls for an umbrella
        },
        "default_favorable_advice": "No umbrella needed; dry skies are expected.",
        "default_unfavorable_advice": "Carry an umbrella or raincoat; precipitation is expected."
    },
    "travel_driving": {
        "key": "travel_driving",
        "domain": "travel_commuting",
        "name": "Travel & Highway Driving",
        "decision_context": "road_safety_and_visibility",
        "aliases_en": [
            r"\btravel\b", r"\btraveling\b", r"\btravelling\b", r"\bdriving\b",
            r"\bdrive\b", r"\broad trip\b", r"\bcommute\b", r"\bcommuting\b",
            r"\bhighway\b", r"\bbike ride\b", r"\btrip\b"
        ],
        "aliases_te": [
            "ప్రయాణం", "డ్రైవింగ్", "రోడ్ ట్రిప్", "బైక్ ప్రయాణం", "వెళ్లవచ్చా",
            "prayaanam", "driving", "travel"
        ],
        "required_variables": ["precipitation", "wind_speed", "weather_condition"],
        "thresholds": {
            "heavy_rain_mm": 20.0,       # Waterlogging & low visibility
            "high_wind_kmh": 35.0        # Two-wheeler instability
        },
        "default_favorable_advice": "Driving and travel conditions are clear and favorable.",
        "default_unfavorable_advice": "Drive with caution; wet roads and reduced visibility may slow traffic."
    },

    # -------------------------------------------------------------------------
    # 3. OUTDOOR WORK, RECREATION & EVENTS
    # -------------------------------------------------------------------------
    "outdoor_event": {
        "key": "outdoor_event",
        "domain": "outdoor_activities",
        "name": "Outdoor Events & Functions",
        "decision_context": "event_weather_suitability",
        "aliases_en": [
            r"\bevent\b", r"\bfunction\b", r"\bparty\b", r"\bwedding\b",
            r"\bpicnic\b", r"\boutdoor gathering\b", r"\bmarriage\b"
        ],
        "aliases_te": [
            "వేడుక", "ఫంక్షన్", "పెళ్లి", "పిక్నిక్", "సభ", "వేడుకలు", "function", "event"
        ],
        "required_variables": ["precipitation", "wind_speed", "temperature"],
        "thresholds": {
            "max_rain_mm": 1.0,
            "max_wind_kmh": 25.0,
            "max_temp_c": 38.0
        },
        "default_favorable_advice": "Favorable weather for an outdoor gathering; comfortable temperatures and clear skies.",
        "default_unfavorable_advice": "Have waterproof canopies or an indoor contingency option ready due to weather risks."
    },
    "construction_work": {
        "key": "construction_work",
        "domain": "outdoor_activities",
        "name": "Construction & Outdoor Labor",
        "decision_context": "worksite_safety_and_materials",
        "aliases_en": [
            r"\bconstruction\b", r"\bconcrete\b", r"\bcement\b", r"\broofing\b",
            r"\bpainting\b", r"\boutdoor work\b", r"\blabor\b", r"\blabour\b"
        ],
        "aliases_te": [
            "నిర్మాణం", "భవన నిర్మాణం", "పెయింటింగ్", "సిమెంట్", "కూలి పని",
            "construction", "painting"
        ],
        "required_variables": ["precipitation", "wind_speed", "temperature"],
        "thresholds": {
            "max_rain_mm": 1.5,          # Concrete washing & painting ruined
            "max_wind_kmh": 28.0,        # Scaffolding hazard
            "max_temp_c": 40.0           # Worker heat exhaustion
        },
        "default_favorable_advice": "Weather is well-suited for outdoor construction, masonry, and painting.",
        "default_unfavorable_advice": "Avoid concrete pouring, exterior painting, or high-elevation scaffolding work."
    },
    "sports_recreation": {
        "key": "sports_recreation",
        "domain": "outdoor_activities",
        "name": "Sports & Outdoor Recreation",
        "decision_context": "outdoor_play_suitability",
        "aliases_en": [
            r"\bcricket\b", r"\bfootball\b", r"\bsports\b", r"\bmatch\b",
            r"\bgame\b", r"\brunning\b", r"\bjogging\b", r"\bcycling\b"
        ],
        "aliases_te": [
            "ఆటలు", "క్రికెట్", "రన్నింగ్", "వాకింగ్", "జాగింగ్", "sports", "cricket"
        ],
        "required_variables": ["precipitation", "temperature", "wind_speed"],
        "thresholds": {
            "max_rain_mm": 0.5,
            "max_wind_kmh": 25.0,
            "max_temp_c": 38.0
        },
        "default_favorable_advice": "Great weather for outdoor sports and recreational activities.",
        "default_unfavorable_advice": "Outdoor play may be interrupted by rain or adverse ground conditions."
    },

    # -------------------------------------------------------------------------
    # 4. HEALTH & ENVIRONMENTAL EXPOSURE
    # -------------------------------------------------------------------------
    "heat_exposure": {
        "key": "heat_exposure",
        "domain": "health_safety",
        "name": "Heatwave & Sun Exposure",
        "decision_context": "thermal_comfort_and_hydration",
        "aliases_en": [
            r"\bheat\b", r"\bheatwave\b", r"\bhot\b", r"\bsun\b",
            r"\bheatstroke\b", r"\btoo hot\b", r"\bsunstroke\b"
        ],
        "aliases_te": [
            "ఎండ", "వేడి", "వడదెబ్బ", "ఎండ తీవ్రత", "ఉక్కపోత", "vada debba", "heat"
        ],
        "required_variables": ["temperature", "humidity"],
        "thresholds": {
            "caution_temp_c": 35.0,
            "extreme_temp_c": 40.0
        },
        "default_favorable_advice": "Temperatures are within comfortable bounds; normal hydration is sufficient.",
        "default_unfavorable_advice": "High heat risk. Stay hydrated and avoid strenuous direct sunlight between 11 AM and 4 PM."
    },
    "cold_exposure": {
        "key": "cold_exposure",
        "domain": "health_safety",
        "name": "Cold Wave & Exposure",
        "decision_context": "hypothermia_and_warmth",
        "aliases_en": [
            r"\bcold\b", r"\bcold wave\b", r"\bchilly\b", r"\bfreezing\b"
        ],
        "aliases_te": [
            "చలి", "చలి తీవ్రత", "చల్లగా", "chali", "cold"
        ],
        "required_variables": ["temperature", "wind_speed"],
        "thresholds": {
            "caution_temp_c": 12.0,
            "extreme_temp_c": 7.0
        },
        "default_favorable_advice": "Comfortable, mild temperatures.",
        "default_unfavorable_advice": "Chilly conditions. Keep warm outerwear on hand, especially for morning and late night."
    },
    "wind_activity": {
        "key": "wind_activity",
        "domain": "outdoor_activities",
        "name": "Wind & Aviation Activities",
        "decision_context": "wind_shear_and_drift",
        "aliases_en": [
            r"\bdrone\b", r"\bkite\b", r"\bflying a kite\b", r"\bboating\b",
            r"\bfishing\b", r"\bhigh wind activity\b", r"\bgale hazard\b"
        ],
        "aliases_te": [
            "గాలిపటం", "పడవ", "డ్రోన్", "చేపల వేట", "gaalipatam"
        ],
        "required_variables": ["wind_speed"],
        "thresholds": {
            "max_wind_kmh": 22.0
        },
        "default_favorable_advice": "Gentle winds suitable for drone flights, light sailing, and rooftop work.",
        "default_unfavorable_advice": "Strong winds may destabilize drones, small boats, or elevated equipment."
    },
    "outdoor_general": {
        "key": "outdoor_general",
        "domain": "outdoor_activities",
        "name": "Outdoor Activities & Going Outside",
        "decision_context": "outdoor_weather_safety",
        "aliases_en": [
            r"\bgo outside\b", r"\bgoing outside\b", r"\bstep outside\b",
            r"\bgo out\b", r"\bgoing out\b", r"\bgo doot\b", r"\bgoing doot\b",
            r"\bcan i go out\b", r"\bcan i go outside\b", r"\boutdoor\b", r"\boutdoors\b"
        ],
        "aliases_te": [
            "బయటకు వెళ్ల", "బయటకు వెళ్లవచ్చా", "బయటికి వెళ్లొచ్చా", "బయటకు", "బయటికి", "bayatiki", "bayataki"
        ],
        "required_variables": ["precipitation", "temperature", "wind_speed", "weather_condition"],
        "thresholds": {
            "max_rain_mm": 0.5,
            "max_wind_kmh": 28.0,
            "max_temp_c": 39.0
        },
        "default_favorable_advice": "Weather conditions are pleasant and safe for going outside.",
        "default_unfavorable_advice": "Consider staying indoors or taking precautions due to unfavorable weather conditions."
    }
}


def find_matching_activity(text: str) -> Optional[Dict[str, Any]]:
    """
    Scans a user query for purpose/activity intents across English and Telugu.
    Returns the matching activity metadata dict if detected, else None.
    """
    if not text or not str(text).strip():
        return None

    raw_text = str(text).strip()
    text_lower = " " + " ".join(raw_text.lower().split()) + " "

    for key, activity in ACTIVITY_REGISTRY.items():
        # 1. Match Telugu aliases (must be at least 3 characters to avoid false hits)
        for te_token in activity.get("aliases_te", []):
            if len(te_token) >= 3 and (te_token in raw_text or te_token.lower() in text_lower):
                return activity

        # 2. Match English aliases (regex / word-boundary)
        for pattern in activity.get("aliases_en", []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                return activity

    return None


def get_required_variables(activity_key: str) -> List[str]:
    """Returns the meteorological variables needed to evaluate the activity."""
    activity = ACTIVITY_REGISTRY.get(activity_key)
    if not activity:
        return ["temperature", "humidity", "wind_speed", "precipitation"]
    return activity.get("required_variables", ["temperature", "precipitation", "wind_speed"])
