# =====================================================
# ENVIRONMENT GUARD - see identical note in backend/main.py.
# Kept here too so this module is safe to import/run
# standalone (e.g. "python -m backend.tools.rag_tool"),
# not just through the FastAPI app.
# =====================================================

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Guard against native access violation in PyTorch CUDA stream capture on CPU/Windows
import torch
if hasattr(torch, "cuda"):
    torch.cuda.is_current_stream_capturing = lambda: False
    if hasattr(torch.cuda, "graphs"):
        torch.cuda.graphs.is_current_stream_capturing = lambda: False

try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_cuda_stream_capturing = lambda: False
except Exception:
    pass

from pathlib import Path
import json
import re
from datetime import datetime,timedelta,date

import faiss
from sentence_transformers import SentenceTransformer


# =====================================================
# PATHS
# =====================================================

BASE_DIR=Path(__file__).resolve().parents[2]

INDEX_DIR=BASE_DIR/"rag"/"index"

INDEX_PATH=INDEX_DIR/"weather_index.faiss"
METADATA_PATH=INDEX_DIR/"metadata.json"


# =====================================================
# SETTINGS
# =====================================================

MODEL_NAME="all-MiniLM-L6-v2"

DEFAULT_TOP_K=5

# Retrieve many candidates first
SEARCH_K=30

MIN_SCORE=0.20


# =====================================================
# RECENCY
# =====================================================

LATEST_DOCUMENT_BOOST=0.10

RECENCY_STEP=0.02

MAX_RECENCY_LEVELS=5


# =====================================================
# KEYWORDS
# =====================================================

KEYWORD_BOOST_PER_MATCH=0.015

MAX_KEYWORD_BOOST=0.10


# =====================================================
# TOPICS
# =====================================================

TOPIC_BOOST_PER_MATCH=0.02

MAX_TOPIC_BOOST=0.10


# =====================================================
# TARGET DATE
# =====================================================

# Exact forecast date match is very important.
TARGET_DATE_BOOST=0.12

# If chunk explicitly contains Day N matching
# the requested forecast date.
DAY_NUMBER_BOOST=0.04


# =====================================================
# LATEST DATE TEXT
# =====================================================

LATEST_DATE_TEXT_BOOST=0.04


# =====================================================
# WARNING CONTENT
# =====================================================

WARNING_CONTENT_BOOST=0.08


# =====================================================
# FOOTER PENALTY
# =====================================================

# IMD PDFs contain repeated footer text.
# Prevent those chunks from ranking too highly.
FOOTER_PENALTY=0.06


# =====================================================
# IMPACT PENALTY
# =====================================================

IMPACT_PENALTY=0.05


# =====================================================
# LAZY-LOADED RAG STATE
#
# The embedding model, FAISS index and metadata used to
# load eagerly at import time. That meant:
#   1. Any failure (missing files, or a native crash from
#      loading torch/faiss/sentence-transformers together)
#      took the whole FastAPI app down at startup, before
#      a single request could be served.
#   2. A native crash (as opposed to a Python exception)
#      could not be caught by orchestrator.py's try/except
#      around search_weather(), which is what was silently
#      killing /chat requests for RAG.
#
# Now these load once, lazily, on first RAG request, inside
# _ensure_rag_loaded(). Genuine Python-level failures (missing
# index/metadata files, bad JSON, etc.) now raise a normal
# exception that orchestrator.py catches and returns as JSON.
# =====================================================

model=None
index=None
metadata=None


def _get_rag_device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _ensure_rag_loaded():

    global model,index,metadata

    if (
        model is not None
        and index is not None
        and metadata is not None
    ):

        return


    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )


    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )


    print("Loading FAISS index...")

    index=faiss.read_index(
        str(INDEX_PATH)
    )


    print("Loading metadata...")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata=json.load(f)


    print("Loading embedding model...")

    model=SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )


    print(
        f"Loaded {index.ntotal} vectors"
    )

    print(
        f"Loaded {len(metadata)} metadata entries"
    )

    global LATEST_DOCUMENT_DATE

    LATEST_DOCUMENT_DATE=get_latest_document_date()

    if LATEST_DOCUMENT_DATE:

        print(
            "Latest IMD document date:",
            LATEST_DOCUMENT_DATE
        )

    else:

        print(
            "Latest IMD document date: Not detected"
        )

# =====================================================
# DATE EXTRACTION FROM SOURCE
# =====================================================

def extract_document_date(source):

    if not source:

        return None

    source=str(source)


    # DD-MM-YYYY
    match=re.search(
        r"(\d{2})-(\d{2})-(\d{4})",
        source
    )

    if match:

        day=int(match.group(1))
        month=int(match.group(2))
        year=int(match.group(3))

        return date(
            year,
            month,
            day
        )


    # DD/MM/YYYY
    match=re.search(
        r"(\d{2})/(\d{2})/(\d{4})",
        source
    )

    if match:

        day=int(match.group(1))
        month=int(match.group(2))
        year=int(match.group(3))

        return date(
            year,
            month,
            day
        )


    # YYYY-MM-DD
    match=re.search(
        r"(\d{4})-(\d{2})-(\d{2})",
        source
    )

    if match:

        year=int(match.group(1))
        month=int(match.group(2))
        day=int(match.group(3))

        return date(
            year,
            month,
            day
        )


    # YYYY_MM_DD
    match=re.search(
        r"(\d{4})_(\d{2})_(\d{2})",
        source
    )

    if match:

        year=int(match.group(1))
        month=int(match.group(2))
        day=int(match.group(3))

        return date(
            year,
            month,
            day
        )


    return None


# =====================================================
# FIND LATEST DOCUMENT
# =====================================================

def get_latest_document_date():

    dates=[]

    for item in metadata:

        source=item.get(
            "source",
            ""
        )

        document_date=extract_document_date(
            source
        )

        if document_date is not None:

            dates.append(
                document_date
            )


    if not dates:

        return None


    return max(
        dates
    )


# Computed lazily inside _ensure_rag_loaded() on first
# RAG request, not at import time (see note above).
LATEST_DOCUMENT_DATE=None


# =====================================================
# CURRENT DATE
# =====================================================

def get_current_date():

    return date.today()


# =====================================================
# QUERY TARGET DATE
# =====================================================

def get_query_target_date(query):

    if not query:

        return None


    query_lower=query.lower()


    current_date=get_current_date()


    # -------------------------------------------------
    # TODAY
    # -------------------------------------------------

    if re.search(
        r"\btoday\b",
        query_lower
    ):

        return current_date


    # -------------------------------------------------
    # TOMORROW
    # -------------------------------------------------

    if re.search(
        r"\btomorrow\b",
        query_lower
    ):

        return current_date+timedelta(days=1)


    # -------------------------------------------------
    # YESTERDAY
    # -------------------------------------------------

    if re.search(
        r"\byesterday\b",
        query_lower
    ):

        return current_date-timedelta(days=1)


    # -------------------------------------------------
    # EXPLICIT DD MONTH
    # -------------------------------------------------

    months={
        "january":1,
        "february":2,
        "march":3,
        "april":4,
        "may":5,
        "june":6,
        "july":7,
        "august":8,
        "september":9,
        "october":10,
        "november":11,
        "december":12
    }


    for month_name,month_number in months.items():

        pattern=(
            r"\b"
            r"(\d{1,2})"
            r"\s+"
            +month_name+
            r"\b"
        )

        match=re.search(
            pattern,
            query_lower
        )

        if match:

            day_number=int(
                match.group(1)
            )

            year=current_date.year

            try:

                return date(
                    year,
                    month_number,
                    day_number
                )

            except ValueError:

                return None


    # -------------------------------------------------
    # EXPLICIT YYYY-MM-DD
    # -------------------------------------------------

    match=re.search(
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        query_lower
    )

    if match:

        try:

            return date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3))
            )

        except ValueError:

            return None


    # -------------------------------------------------
    # EXPLICIT DD-MM-YYYY
    # -------------------------------------------------

    match=re.search(
        r"\b(\d{2})-(\d{2})-(\d{4})\b",
        query_lower
    )

    if match:

        try:

            return date(
                int(match.group(3)),
                int(match.group(2)),
                int(match.group(1))
            )

        except ValueError:

            return None


    return None


# =====================================================
# EXTRACT FORECAST DATES FROM CHUNK
# =====================================================

def extract_forecast_dates(text):

    if not text:

        return []


    text_lower=text.lower()

    months={
        "january":1,
        "february":2,
        "march":3,
        "april":4,
        "may":5,
        "june":6,
        "july":7,
        "august":8,
        "september":9,
        "october":10,
        "november":11,
        "december":12
    }


    dates=[]


    # -------------------------------------------------
    # DD MONTH
    # -------------------------------------------------

    for month_name,month_number in months.items():

        pattern=(
            r"\b"
            r"(\d{1,2})"
            r"\s+"
            +month_name+
            r"\b"
        )

        matches=re.finditer(
            pattern,
            text_lower
        )

        for match in matches:

            day_number=int(
                match.group(1)
            )

            # Use current year.
            # IMD documents here are current-year bulletins.
            year=get_current_date().year

            try:

                dates.append(
                    date(
                        year,
                        month_number,
                        day_number
                    )
                )

            except ValueError:

                pass


    # -------------------------------------------------
    # YYYY-MM-DD
    # -------------------------------------------------

    matches=re.finditer(
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        text_lower
    )

    for match in matches:

        try:

            dates.append(
                date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3))
                )
            )

        except ValueError:

            pass


    # -------------------------------------------------
    # DD-MM-YYYY
    # -------------------------------------------------

    matches=re.finditer(
        r"\b(\d{2})-(\d{2})-(\d{4})\b",
        text_lower
    )

    for match in matches:

        try:

            dates.append(
                date(
                    int(match.group(3)),
                    int(match.group(2)),
                    int(match.group(1))
                )
            )

        except ValueError:

            pass


    return list(
        dict.fromkeys(dates)
    )


# =====================================================
# EXTRACT DAY NUMBER
# =====================================================

def extract_day_number(text):

    if not text:

        return None


    match=re.search(
        r"\bday\s*(\d+)\b",
        text.lower()
    )

    if not match:

        return None


    return int(
        match.group(1)
    )


# =====================================================
# DETERMINE DAY NUMBER FOR TARGET DATE
# =====================================================

def calculate_expected_day_number(
    target_date,
    document_date
):

    if target_date is None:

        return None


    if document_date is None:

        return None


    difference=(
        target_date-document_date
    ).days


    # IMD:
    #
    # Day 1 = document date
    # Day 2 = document date + 1
    # Day 3 = document date + 2
    #
    # Therefore:
    # expected Day = difference + 1

    if 0<=difference<=10:

        return difference+1


    return None


# =====================================================
# RECENCY BOOST
# =====================================================

def calculate_recency_boost(source):

    if LATEST_DOCUMENT_DATE is None:

        return 0.0


    document_date=extract_document_date(
        source
    )


    if document_date is None:

        return 0.0


    days_old=(
        LATEST_DOCUMENT_DATE-document_date
    ).days


    if days_old<0:

        days_old=0


    level=min(
        days_old,
        MAX_RECENCY_LEVELS
    )


    boost=(
        LATEST_DOCUMENT_BOOST
        -(level*RECENCY_STEP)
    )


    return max(
        0.0,
        boost
    )


# =====================================================
# QUERY KEYWORDS
# =====================================================

def extract_keywords(query):

    if not query:

        return []


    words=re.findall(
        r"[a-zA-Z]+",
        query.lower()
    )


    stop_words={

        "what",
        "is",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "and",
        "or",
        "with",
        "from",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "today",
        "tomorrow",
        "yesterday",
        "please",
        "can",
        "you",
        "tell",
        "me",
        "about",
        "give",
        "show",
        "latest",
        "current",
        "now"

    }


    keywords=[]


    for word in words:

        if len(word)<3:

            continue


        if word in stop_words:

            continue


        if word not in keywords:

            keywords.append(
                word
            )


    return keywords


# =====================================================
# KEYWORD BOOST
# =====================================================

def calculate_keyword_boost(
    query,
    text
):

    if not query or not text:

        return 0.0


    keywords=extract_keywords(
        query
    )


    if not keywords:

        return 0.0


    text_lower=text.lower()

    matched=0


    for keyword in keywords:

        pattern=(
            r"\b"
            +re.escape(keyword)
            +r"\b"
        )


        if re.search(
            pattern,
            text_lower
        ):

            matched+=1


    boost=(
        matched
        *KEYWORD_BOOST_PER_MATCH
    )


    return min(
        boost,
        MAX_KEYWORD_BOOST
    )


# =====================================================
# TOPIC BOOST
# =====================================================

def calculate_topic_boost(
    query,
    text
):

    if not query or not text:

        return 0.0


    query_lower=query.lower()
    text_lower=text.lower()


    topic_groups={

        "rainfall":[
            "rainfall",
            "rain",
            "heavy rainfall",
            "very heavy rainfall",
            "extremely heavy rainfall",
            "precipitation"
        ],

        "warning":[
            "warning",
            "red color warning",
            "orange color warning",
            "yellow color warning"
        ],

        "temperature":[
            "temperature",
            "maximum temperature",
            "minimum temperature",
            "max temp",
            "min temp"
        ],

        "thunderstorm":[
            "thunderstorm",
            "thunder",
            "lightning"
        ],

        "forecast":[
            "forecast",
            "weather forecast"
        ]

    }


    boost=0.0


    for terms in topic_groups.values():

        query_has_topic=False
        document_has_topic=False


        for term in terms:

            if term in query_lower:

                query_has_topic=True


            if term in text_lower:

                document_has_topic=True


        if (
            query_has_topic
            and document_has_topic
        ):

            boost+=TOPIC_BOOST_PER_MATCH


    return min(
        boost,
        MAX_TOPIC_BOOST
    )


# =====================================================
# TARGET DATE BOOST
# =====================================================

def calculate_target_date_boost(
    query,
    text,
    source
):

    target_date=get_query_target_date(
        query
    )


    if target_date is None:

        return 0.0


    forecast_dates=extract_forecast_dates(
        text
    )


    if not forecast_dates:

        return 0.0


    if target_date in forecast_dates:

        return TARGET_DATE_BOOST


    return 0.0


# =====================================================
# DAY NUMBER BOOST
# =====================================================

def calculate_day_number_boost(
    query,
    text,
    source
):

    target_date=get_query_target_date(
        query
    )


    if target_date is None:

        return 0.0


    document_date=extract_document_date(
        source
    )


    if document_date is None:

        return 0.0


    expected_day=calculate_expected_day_number(
        target_date,
        document_date
    )


    if expected_day is None:

        return 0.0


    actual_day=extract_day_number(
        text
    )


    if actual_day is None:

        return 0.0


    if actual_day==expected_day:

        return DAY_NUMBER_BOOST


    return 0.0


# =====================================================
# LATEST DATE TEXT BOOST
# =====================================================

def calculate_latest_date_text_boost(
    text
):

    if not LATEST_DOCUMENT_DATE:

        return 0.0


    year=LATEST_DOCUMENT_DATE.year
    month=LATEST_DOCUMENT_DATE.month
    day=LATEST_DOCUMENT_DATE.day


    month_name=LATEST_DOCUMENT_DATE.strftime(
        "%B"
    )


    date_patterns=[

        f"{year}-{month:02d}-{day:02d}",

        f"{day:02d}-{month:02d}-{year}",

        f"{day} {month_name}",

        f"{day}th {month_name}",

        f"{day}st {month_name}",

        f"{day}nd {month_name}",

        f"{day}rd {month_name}"

    ]


    text_lower=text.lower()


    for pattern in date_patterns:

        if pattern.lower() in text_lower:

            return LATEST_DATE_TEXT_BOOST


    return 0.0


# =====================================================
# WARNING CONTENT BOOST
# =====================================================

def calculate_warning_content_boost(
    query,
    text
):

    if not query or not text:

        return 0.0


    query_lower=query.lower()
    text_lower=text.lower()


    warning_query_terms=[

        "warning",
        "rainfall warning",
        "rain warning",
        "heavy rain",
        "rainfall",
        "precipitation warning"

    ]


    query_is_warning=any(
        term in query_lower
        for term in warning_query_terms
    )


    if not query_is_warning:

        return 0.0


    warning_terms=[

        "heavy rainfall",
        "very heavy rainfall",
        "extremely heavy rainfall",
        "heavy to very heavy rainfall",
        "warning",
        "very likely at isolated places",
        "likely at isolated places"

    ]


    matched=False


    for term in warning_terms:

        if term in text_lower:

            matched=True
            break


    if matched:

        return WARNING_CONTENT_BOOST


    return 0.0


# =====================================================
# FOOTER PENALTY
# =====================================================

def calculate_footer_penalty(
    text
):

    if not text:

        return 0.0


    text_lower=text.lower()


    footer_terms=[

        "for more details kindly visit",

        "contact 011-2434-4599",

        "service to the nation since 1875",

        "forecast and warning for any day is valid",

        "red color warning does not mean"

    ]


    matches=0


    for term in footer_terms:

        if term in text_lower:

            matches+=1


    if matches>=2:

        return FOOTER_PENALTY


    return 0.0


# =====================================================
# IMPACT PENALTY
# =====================================================

def calculate_impact_penalty(
    query,
    text
):

    query_lower=query.lower()
    text_lower=text.lower()


    warning_query=(

        "warning" in query_lower

        or "rainfall" in query_lower

        or "rain" in query_lower

    )


    if not warning_query:

        return 0.0


    impact_terms=[

        "impact expected",

        "action suggested",

        "impact and actions",

        "expected impact"

    ]


    for term in impact_terms:

        if term in text_lower:

            return IMPACT_PENALTY


    return 0.0


# =====================================================
# QUERY INTENT
# =====================================================

def get_query_intent(query):

    query_lower=query.lower()


    target_date=get_query_target_date(
        query
    )


    asks_warning=any(

        term in query_lower

        for term in [

            "warning",
            "rainfall warning",
            "rain warning"

        ]

    )


    asks_rain=any(

        term in query_lower

        for term in [

            "rain",
            "rainfall",
            "precipitation"

        ]

    )


    asks_forecast=any(

        term in query_lower

        for term in [

            "forecast",
            "weather"

        ]

    )


    asks_latest=any(

        term in query_lower

        for term in [

            "latest",
            "current",
            "now"

        ]

    )


    return {

        "target_date":target_date,

        "asks_warning":asks_warning,

        "asks_rain":asks_rain,

        "asks_forecast":asks_forecast,

        "asks_latest":asks_latest

    }


# =====================================================
# RETRIEVE WEATHER CONTEXT
# =====================================================

def retrieve_weather_context(
    query,
    top_k=DEFAULT_TOP_K
):
    _ensure_rag_loaded()


    if not query or not query.strip():

        return []


    if index.ntotal<=0:

        return []


    # -------------------------------------------------
    # QUERY INTENT
    # -------------------------------------------------

    intent=get_query_intent(
        query
    )


    target_date=intent[
        "target_date"
    ]


    # -------------------------------------------------
    # SEARCH K
    # -------------------------------------------------

    search_k=min(
        SEARCH_K,
        index.ntotal
    )


    # -------------------------------------------------
    # EMBEDDING
    # -------------------------------------------------

    query_embedding=model.encode(
        [query],
        convert_to_numpy=True
    )


    query_embedding=query_embedding.astype(
        "float32"
    )


    # -------------------------------------------------
    # NORMALIZE
    # -------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )


    # -------------------------------------------------
    # FAISS SEARCH
    # -------------------------------------------------

    scores,indices=index.search(
        query_embedding,
        search_k
    )


    candidates=[]


    # =================================================
    # PROCESS CANDIDATES
    # =================================================

    for score,idx in zip(
        scores[0],
        indices[0]
    ):

        if idx<0:

            continue


        if idx>=len(metadata):

            continue


        faiss_score=float(
            score
        )


        if faiss_score<MIN_SCORE:

            continue


        item=metadata[idx]


        source=item.get(
            "source",
            "Unknown"
        )


        text=item.get(
            "text",
            ""
        )


        # -------------------------------------------------
        # DOCUMENT DATE
        # -------------------------------------------------

        document_date=extract_document_date(
            source
        )


        # -------------------------------------------------
        # BOOSTS
        # -------------------------------------------------

        recency_boost=calculate_recency_boost(
            source
        )


        keyword_boost=calculate_keyword_boost(
            query,
            text
        )


        topic_boost=calculate_topic_boost(
            query,
            text
        )


        target_date_boost=calculate_target_date_boost(
            query,
            text,
            source
        )


        day_number_boost=calculate_day_number_boost(
            query,
            text,
            source
        )


        latest_date_text_boost=calculate_latest_date_text_boost(
            text
        )


        warning_content_boost=calculate_warning_content_boost(
            query,
            text
        )


        footer_penalty=calculate_footer_penalty(
            text
        )


        impact_penalty=calculate_impact_penalty(
            query,
            text
        )


        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        final_score=(

            faiss_score

            +recency_boost

            +keyword_boost

            +topic_boost

            +target_date_boost

            +day_number_boost

            +latest_date_text_boost

            +warning_content_boost

            -impact_penalty

            -footer_penalty

        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        candidates.append({

            "source":source,

            "chunk_id":item.get(
                "chunk_id",
                -1
            ),

            "score":round(
                final_score,
                4
            ),

            "faiss_score":round(
                faiss_score,
                4
            ),

            "recency_boost":round(
                recency_boost,
                4
            ),

            "keyword_boost":round(
                keyword_boost,
                4
            ),

            "topic_boost":round(
                topic_boost,
                4
            ),

            "target_date_boost":round(
                target_date_boost,
                4
            ),

            "day_number_boost":round(
                day_number_boost,
                4
            ),

            "latest_date_text_boost":round(
                latest_date_text_boost,
                4
            ),

            "warning_content_boost":round(
                warning_content_boost,
                4
            ),

            "impact_penalty":round(
                impact_penalty,
                4
            ),

            "footer_penalty":round(
                footer_penalty,
                4
            ),

            "document_date":(
                document_date.isoformat()
                if document_date
                else None
            ),

            "forecast_dates":[
                d.isoformat()
                for d in extract_forecast_dates(
                    text
                )
            ],

            "day_number":extract_day_number(
                text
            ),

            "text":text

        })


    # =================================================
    # SORT
    # =================================================

    candidates.sort(

        key=lambda x:(

            x["score"],

            x["faiss_score"]

        ),

        reverse=True

    )


    # =================================================
    # RETURN
    # =================================================

    return candidates[:top_k]


# =====================================================
# SEARCH WEATHER
# =====================================================

def search_weather(
    query,
    top_k=DEFAULT_TOP_K
):

    return retrieve_weather_context(
        query,
        top_k
    )


# =====================================================
# BUILD RAG CONTEXT
# =====================================================

def get_context(
    query,
    top_k=DEFAULT_TOP_K
):

    results=retrieve_weather_context(
        query,
        top_k
    )


    if not results:

        return (

            "No relevant weather information "
            "was found in the RAG documents."

        )


    context=[]


    for i,result in enumerate(
        results,
        1
    ):

        context.append(

            f"""SOURCE {i}

DOCUMENT: {result['source']}

CHUNK: {result['chunk_id']}

FINAL SCORE: {result['score']}

FAISS SCORE: {result['faiss_score']}

DOCUMENT DATE: {result['document_date']}

FORECAST DATES: {result['forecast_dates']}

DAY NUMBER: {result['day_number']}

RECENCY BOOST: {result['recency_boost']}

KEYWORD BOOST: {result['keyword_boost']}

TOPIC BOOST: {result['topic_boost']}

TARGET DATE BOOST: {result['target_date_boost']}

DAY NUMBER BOOST: {result['day_number_boost']}

LATEST DATE TEXT BOOST: {result['latest_date_text_boost']}

WARNING CONTENT BOOST: {result['warning_content_boost']}

IMPACT PENALTY: {result['impact_penalty']}

FOOTER PENALTY: {result['footer_penalty']}

{result['text']}"""

        )


    return "\n\n".join(
        context
    )


# =====================================================
# RAG INFORMATION
# =====================================================

def get_rag_info():

    try:
        _ensure_rag_loaded()
    except Exception as e:
        return {
            "provider":"FAISS",
            "embedding_model":MODEL_NAME,
            "index":"weather_index.faiss",
            "available":False,
            "error":str(e)
        }

    return {

        "provider":"FAISS",

        "embedding_model":MODEL_NAME,

        "index":"weather_index.faiss",

        "vectors":index.ntotal if index else 0,

        "metadata_entries":len(metadata) if metadata else 0,

        "top_k":DEFAULT_TOP_K,

        "search_k":SEARCH_K,

        "minimum_score":MIN_SCORE,

        "latest_document_date":(

            LATEST_DOCUMENT_DATE.isoformat()

            if LATEST_DOCUMENT_DATE

            else None

        ),

        "latest_document_boost":
            LATEST_DOCUMENT_BOOST,

        "keyword_boost_per_match":
            KEYWORD_BOOST_PER_MATCH,

        "maximum_keyword_boost":
            MAX_KEYWORD_BOOST,

        "topic_boost_per_match":
            TOPIC_BOOST_PER_MATCH,

        "target_date_boost":
            TARGET_DATE_BOOST,

        "day_number_boost":
            DAY_NUMBER_BOOST,

        "latest_date_text_boost":
            LATEST_DATE_TEXT_BOOST,

        "warning_content_boost":
            WARNING_CONTENT_BOOST,

        "impact_penalty":
            IMPACT_PENALTY,

        "footer_penalty":
            FOOTER_PENALTY

    }


# =====================================================
# DIRECT TEST
# =====================================================

if __name__=="__main__":

    print()
    print("================================")
    print("WeatherGPT+ RAG Search")
    print("================================")
    print()

    print(
        "Latest document:",
        LATEST_DOCUMENT_DATE
    )

    print(
        "System date:",
        get_current_date()
    )


    query=input(
        "\nAsk a weather question: "
    )


    print()
    print(
        "Target forecast date:",
        get_query_target_date(query)
    )


    results=search_weather(
        query
    )


    print()
    print("================================")
    print("RETRIEVED RESULTS")
    print("================================")


    if not results:

        print(
            "No relevant results found."
        )

    else:

        for i,result in enumerate(
            results,
            1
        ):

            print(
                f"\n--- RESULT {i} ---"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Final Score: {result['score']}"
            )

            print(
                f"FAISS Score: {result['faiss_score']}"
            )

            print(
                f"Document Date: {result['document_date']}"
            )

            print(
                f"Forecast Dates: {result['forecast_dates']}"
            )

            print(
                f"Day Number: {result['day_number']}"
            )

            print(
                f"Recency Boost: {result['recency_boost']}"
            )

            print(
                f"Keyword Boost: {result['keyword_boost']}"
            )

            print(
                f"Topic Boost: {result['topic_boost']}"
            )

            print(
                f"Target Date Boost: {result['target_date_boost']}"
            )

            print(
                f"Day Number Boost: {result['day_number_boost']}"
            )

            print(
                f"Latest Date Text Boost: "
                f"{result['latest_date_text_boost']}"
            )

            print(
                f"Warning Content Boost: "
                f"{result['warning_content_boost']}"
            )

            print(
                f"Impact Penalty: "
                f"{result['impact_penalty']}"
            )

            print(
                f"Footer Penalty: "
                f"{result['footer_penalty']}"
            )

            print(
                f"\n{result['text']}"
            )