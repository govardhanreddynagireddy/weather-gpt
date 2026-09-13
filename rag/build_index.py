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

DOCUMENTS_DIR.mkdir(exist_ok=True)
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

    for page_number,page in enumerate(reader.pages,1):

        try:

            page_text=page.extract_text()

            if page_text:

                text+=page_text+"\n"

        except Exception as e:

            print(
                f"  Warning: could not read page {page_number}: {e}"
            )

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

    pdf_files=sorted(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        text=extract_text(pdf_path)

        print(
            f"  Extracted characters: {len(text)}"
        )

        if not text.strip():

            print("  WARNING: No text extracted")

            continue

        chunks=create_chunks(text)

        print(
            f"  Created chunks: {len(chunks)}"
        )

        for chunk_number,chunk in enumerate(chunks):

            all_chunks.append(chunk)

            metadata.append({

                "source":pdf_path.name,

                "chunk_id":chunk_number,

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

        show_progress_bar=True,

        batch_size=32

    )

    # Convert to float32 for FAISS
    embeddings=embeddings.astype("float32")

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dimension=embeddings.shape[1]

    index=faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"\nEmbedding dimension: {dimension}"
    )

    print(
        f"Vectors stored: {index.ntotal}"
    )

    return index


# -----------------------------
# Save index
# -----------------------------

def save_index(index,metadata):

    index_path=INDEX_DIR/"weather_index.faiss"

    metadata_path=INDEX_DIR/"metadata.json"

    faiss.write_index(
        index,
        str(index_path)
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nFiles saved:")

    print(
        f"FAISS index: {index_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


# -----------------------------
# Main
# -----------------------------

def main():

    print("========================================")
    print("WeatherGPT+ RAG Index Builder")
    print("========================================")

    print(
        f"\nDocuments directory:\n{DOCUMENTS_DIR}"
    )

    chunks,metadata=load_documents()

    if not chunks:

        print(
            "\nNo text found in the PDF documents."
        )

        return

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    index=build_index(chunks)

    save_index(
        index,
        metadata
    )

    print("\n========================================")
    print("RAG index created successfully!")
    print("========================================")


if __name__=="__main__":

    main()