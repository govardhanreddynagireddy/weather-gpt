from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR=Path(__file__).resolve().parent
INDEX_DIR=BASE_DIR/"index"

INDEX_PATH=INDEX_DIR/"weather_index.faiss"
METADATA_PATH=INDEX_DIR/"metadata.json"

MODEL_NAME="all-MiniLM-L6-v2"


model=SentenceTransformer(MODEL_NAME)

index=faiss.read_index(str(INDEX_PATH))

with open(METADATA_PATH,"r",encoding="utf-8") as f:
    metadata=json.load(f)


def retrieve_weather_context(query,top_k=5):

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
            "source":metadata[idx]["source"],
            "score":round(float(score),4),
            "text":metadata[idx]["text"]
        })

    return results


def get_context(query,top_k=5):

    results=retrieve_weather_context(query,top_k)

    context=""

    for i,result in enumerate(results,1):

        context+=f"""
SOURCE {i}: {result['source']}
RELEVANCE: {result['score']}

{result['text']}

"""

    return context


if __name__=="__main__":

    query=input("Ask a weather question: ")

    context=get_context(query)

    print("\n==============================")
    print("RETRIEVED WEATHER CONTEXT")
    print("==============================")

    print(context)