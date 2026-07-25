"""
Loads documents from data/, splits them into overlapping chunks,
embeds them, and stores them in a local Chroma collection.
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between consecutive chunks
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "docs"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += step
    return chunks


def load_documents(data_dir: str = "data") -> list[dict]:
    """Read every .md/.txt file in data_dir into {"source", "text"} dicts."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"source": filename, "text": text})
    return documents


def build_index(data_dir: str = "data"):
    """Load documents, chunk them, embed the chunks, and upsert into Chroma."""
    documents = load_documents(data_dir)
    if not documents:
        print(f"No documents found in '{data_dir}'. Add .md/.txt files and re-run.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_ids, all_texts, all_metadatas = [], [], []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{doc['source']}::{i}")
            all_texts.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})

    embeddings = model.encode(all_texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
    )

    print(f"Indexed {len(all_texts)} chunks from {len(documents)} documents into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()
