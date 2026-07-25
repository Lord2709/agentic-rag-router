"""
The retrieve tool: given a query string, returns the top-k most similar
chunks from the Chroma collection, each with metadata and a similarity score.

Design note: this function does NOT decide whether results are "good enough" -
that judgment belongs to the judge tool. This just returns facts: here's what's
closest, and here's how close.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingest import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

TOP_K = 5

# Loaded once, at import time - reused across every retrieve() call rather
# than reloaded per call (the orchestrator will call retrieve() repeatedly
# across retries, so this matters).
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection = _client.get_collection(COLLECTION_NAME)  # NOT get_or_create -
# if this raises, it means ingest.py hasn't been run yet, and that should
# fail loudly rather than silently querying an empty collection.


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Returns a list of dicts: {"text", "source", "chunk_id", "score"}
    where score = 1 - cosine_distance (so 1.0 = identical, 0.0 = unrelated).
    """
    query_embedding = _model.encode([query]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        retrieved.append({
            "text": text,
            "source": metadata["source"],
            "chunk_id": chunk_id,
            "score": 1 - distance,
        })

    return retrieved
