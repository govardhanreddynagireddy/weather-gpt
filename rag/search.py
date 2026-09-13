from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------

BASE_DIR=Path(__file__).resolve().parent
INDEX_DIR=BASE_DIR/"index"

INDEX_PATH=INDEX_DIR/"weather_index.faiss"
METADATA_PATH=INDEX_DIR/"metadata.json"


# -----------------------------
# Embedding model
# -----------------------------

MODEL_NAME="all-MiniLM-L6-v2"

model=SentenceTransformer(MODEL_NAME)


# -----------------------------
# Load FAISS index
# -----------------------------

index=faiss.read_index(str(INDEX_PATH))


# -----------------------------
# Load metadata
# -----------------------------

with open(METADATA_PATH,"r",encoding="utf-8") as f:
    metadata=json.load(f)


# -----------------------------
# Search function
# -----------------------------

def search(query,top_k=5):

    query_embedding=model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    scores,indices=index.search(query_embedding,top_k)

    results=[]

    for score,idx in zip(scores[0],indices[0]):

        if idx==-1:
            continue

        results.append({
            "score":float(score),
            "source":metadata[idx]["source"],
            "text":metadata[idx]["text"]
        })

    return results


# -----------------------------
# Test
# -----------------------------

if __name__=="__main__":

    query=input("\nAsk a weather question: ")

    results=search(query)

    print("\n==============================")
    print("RAG SEARCH RESULTS")
    print("==============================")

    for i,result in enumerate(results,1):

        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text:\n{result['text']}")