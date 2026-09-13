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

print("Loading embedding model...")

model=SentenceTransformer(
    MODEL_NAME
)


# =====================================================
# CHECK FILES
# =====================================================

if not INDEX_PATH.exists():

    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_PATH}\n\n"
        "Run:\n"
        "python rag/build_index.py"
    )


if not METADATA_PATH.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_PATH}\n\n"
        "Run:\n"
        "python rag/build_index.py"
    )


# =====================================================
# LOAD FAISS INDEX
# =====================================================

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
    f"Loaded {index.ntotal} vectors"
)

print(
    f"Loaded {len(metadata)} metadata entries"
)


# =====================================================
# SEARCH
# =====================================================

def search(
    query,
    top_k=DEFAULT_TOP_K
):

    if not query or not query.strip():

        return []


    if index.ntotal==0:

        return []


    # Do not request more results
    # than the index contains.

    search_k=min(
        top_k,
        index.ntotal
    )


    # -------------------------------------------------
    # Convert query to embedding
    # -------------------------------------------------

    query_embedding=model.encode(

        [query],

        convert_to_numpy=True

    )


    # FAISS expects float32

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
    # FAISS search
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


        # Remove very weak matches.

        if score<MIN_SCORE:

            continue


        item=metadata[idx]


        results.append({

            "score":round(
                score,
                4
            ),

            "source":item.get(
                "source",
                "Unknown"
            ),

            "chunk_id":item.get(
                "chunk_id",
                -1
            ),

            "text":item.get(
                "text",
                ""
            )

        })


    return results


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


    results=search(
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
                f"Text:\n{result['text']}"
            )