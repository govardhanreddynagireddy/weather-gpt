from pathlib import Path
import json

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------

BASE_DIR=Path(__file__).resolve().parent

DOCUMENTS_DIR=BASE_DIR/"documents"
INDEX_DIR=BASE_DIR/"index"

INDEX_DIR.mkdir(exist_ok=True)


# -----------------------------
# Settings
# -----------------------------

CHUNK_SIZE=800
CHUNK_OVERLAP=150

MODEL_NAME="all-MiniLM-L6-v2"


# -----------------------------
# Extract text from PDFs
# -----------------------------

def extract_text(pdf_path):

    reader=PdfReader(pdf_path)

    text=""

    for page in reader.pages:

        page_text=page.extract_text()

        if page_text:
            text+=page_text+"\n"

    return text


# -----------------------------
# Split text into chunks
# -----------------------------

def create_chunks(text):

    text=" ".join(text.split())

    chunks=[]

    start=0

    while start<len(text):

        end=start+CHUNK_SIZE

        chunk=text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start=end-CHUNK_OVERLAP

    return chunks


# -----------------------------
# Load all documents
# -----------------------------

def load_documents():

    all_chunks=[]
    metadata=[]

    pdf_files=list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        text=extract_text(pdf_path)

        print(f"  Extracted characters: {len(text)}")

        chunks=create_chunks(text)

        print(f"  Created chunks: {len(chunks)}")

        for chunk in chunks:

            all_chunks.append(chunk)

            metadata.append({
                "source":pdf_path.name,
                "text":chunk
            })

    return all_chunks,metadata


# -----------------------------
# Build FAISS index
# -----------------------------

def build_index(chunks):

    print("\nLoading embedding model...")

    model=SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    embeddings=model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dimension=embeddings.shape[1]

    index=faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"\nEmbedding dimension: {dimension}")
    print(f"Vectors stored: {index.ntotal}")

    return index


# -----------------------------
# Main
# -----------------------------

def main():

    print("================================")
    print("WeatherGPT+ RAG Index Builder")
    print("================================\n")

    chunks,metadata=load_documents()

    if not chunks:
        print("No text found in the PDFs.")
        return

    print(f"\nTotal chunks: {len(chunks)}")

    index=build_index(chunks)

    # Save FAISS index
    index_path=INDEX_DIR/"weather_index.faiss"

    faiss.write_index(index,str(index_path))

    # Save metadata
    metadata_path=INDEX_DIR/"metadata.json"

    with open(metadata_path,"w",encoding="utf-8") as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)

    print("\n================================")
    print("RAG index created successfully!")
    print("================================")

    print(f"FAISS index: {index_path}")
    print(f"Metadata: {metadata_path}")


if __name__=="__main__":
    main()