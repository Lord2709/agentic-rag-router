"""
The retrieve tool: given a query string, returns the top-k most similar
chunks from the Chroma collection, each with metadata and a similarity score.

Design note: this function does NOT decide whether results are "good enough" -
that judgment belongs to the judge tool. This just returns facts: here's what's
closest, and here's how close.
"""

TOP_K = 5


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Args:
        query: the (possibly reformulated) search query
        k: number of chunks to return

    Returns:
        A list of dicts, each shaped like:
        {
            "text": str,        # the chunk's text
            "source": str,      # originating filename
            "chunk_id": str,    # unique id
            "score": float,     # similarity score (higher = more similar,
                                 # or distance, lower = more similar - pick one
                                 # convention and be consistent)
        }

    TODO:
    - embed `query` using the same embedding model used in ingest.py
    - query the Chroma collection for top-k nearest chunks
    - reshape Chroma's raw response (documents, metadatas, distances) into
      the list-of-dicts format above
    - decide: are you exposing Chroma's raw distance, or converting it to a
      0-1 similarity score? (Chroma returns distance by default - smaller
      is more similar. Consider converting to something more intuitive.)
    """
    raise NotImplementedError
