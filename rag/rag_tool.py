from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# =====================================================
# PATHS
# =====================================================

BASE_DIR=Path(__file__).resolve().parent

INDEX_DIR=BASE_DIR/"index"

INDEX_PATH=INDEX_DIR/"weather_index.faiss"
METADATA_PATH=INDEX_DIR/"metadata.json"


# =====================================================
# SETTINGS
# =====================================================

MODEL_NAME="all-MiniLM-L6-v2"

DEFAULT_TOP_K=5

MIN_SCORE=0.20


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading RAG embedding model...")

model=SentenceTransformer(
    MODEL_NAME
)


# =====================================================
# CHECK RAG FILES
# =====================================================

if not INDEX_PATH.exists():

    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_PATH}\n"
        "Run: python rag/build_index.py"
    )


if not METADATA_PATH.exists():

    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}\n"
        "Run: python rag/build_index.py"
    )


# =====================================================
# LOAD FAISS INDEX
# =====================================================

print("Loading FAISS index...")

index=faiss.read_index(
    str(INDEX_PATH)
)


# =====================================================
# LOAD METADATA
# =====================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata=json.load(f)


print(
    f"RAG ready: {index.ntotal} vectors, "
    f"{len(metadata)} metadata entries"
)


# =====================================================
# RETRIEVE WEATHER CONTEXT
# =====================================================

def retrieve_weather_context(
    query,
    top_k=DEFAULT_TOP_K
):

    if not query or not query.strip():

        return []


    # -------------------------------------------------
    # Prevent requesting more vectors than available
    # -------------------------------------------------

    search_k=min(
        top_k,
        index.ntotal
    )


    if search_k==0:

        return []


    # -------------------------------------------------
    # Create query embedding
    # -------------------------------------------------

    query_embedding=model.encode(

        [query],

        convert_to_numpy=True
    )


    # -------------------------------------------------
    # Convert to float32
    # -------------------------------------------------

    query_embedding=query_embedding.astype(
        "float32"
    )


    # -------------------------------------------------
    # Normalize
    # -------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )


    # -------------------------------------------------
    # Search FAISS
    # -------------------------------------------------

    scores,indices=index.search(

        query_embedding,

        search_k
    )


    results=[]


    # -------------------------------------------------
    # Process results
    # -------------------------------------------------

    for score,idx in zip(
        scores[0],
        indices[0]
    ):

        if idx<0:

            continue


        if idx>=len(metadata):

            continue


        score=float(score)


        # -------------------------------------------------
        # Ignore weak matches
        # -------------------------------------------------

        if score<MIN_SCORE:

            continue


        item=metadata[idx]


        results.append({

            "source":item.get(
                "source",
                "Unknown"
            ),

            "chunk_id":item.get(
                "chunk_id",
                -1
            ),

            "score":round(
                score,
                4
            ),

            "text":item.get(
                "text",
                ""
            )

        })


    return results


# =====================================================
# BUILD CONTEXT FOR LLM
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


    context_parts=[]


    for i,result in enumerate(
        results,
        1
    ):

        context_parts.append(

            f"""
SOURCE {i}
DOCUMENT: {result['source']}
CHUNK: {result['chunk_id']}
RELEVANCE: {result['score']}

{result['text']}
""".strip()

        )


    return "\n\n".join(
        context_parts
    )


# =====================================================
# STRUCTURED SEARCH
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
# SIMPLE TEXT SEARCH FOR AGENT
# =====================================================

def search_weather_text(
    query,
    top_k=DEFAULT_TOP_K
):

    return get_context(
        query,
        top_k
    )


# =====================================================
# TEST
# =====================================================

if __name__=="__main__":

    print(
        "\n================================"
    )

    print(
        "WeatherGPT RAG Search"
    )

    print(
        "================================"
    )


    query=input(
        "\nAsk a weather question: "
    )


    results=retrieve_weather_context(
        query
    )


    print(
        "\n=============================="
    )

    print(
        "RAG SEARCH RESULTS"
    )

    print(
        "=============================="
    )


    if not results:

        print(
            "\nNo relevant results found."
        )


    else:

        for i,result in enumerate(
            results,
            1
        ):

            print(
                f"\n--- Result {i} ---"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"\n{result['text']}"
            )