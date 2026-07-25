# Agentic RAG Router

A retrieval-augmented generation system where retrieval is a *decision*, not a fixed step.

Instead of always embedding a query and stuffing top-k chunks into a prompt, this system:

1. **Routes** — decides whether to retrieve at all, and from which source.
2. **Reformulates** — rewrites weak/vague queries before (re)searching.
3. **Judges** — scores retrieved context for relevance before generating.
4. **Retries** — if context is thin, reformulates and searches again (capped).
5. **Falls back** — if retries are exhausted, returns "insufficient information" instead of hallucinating.

## Stack

- LLM: Claude (Haiku for router/judge, Sonnet for reformulate/generate) via Anthropic API, using native tool-calling
- Vector store: Chroma (local, embedded)
- Embeddings: local `sentence-transformers` model (no API cost)
- Corpus: personal docs in `data/`

## Project layout

```
src/
  config.py          # env/config loading
  ingest.py           # load + chunk + embed docs into Chroma
  tools/
    retrieve.py       # vector search tool
    reformulate.py    # query rewriting tool
    judge.py          # relevance gate tool
    generate.py       # final answer generation
  orchestrator.py     # LLM tool-calling loop tying it all together
  main.py             # CLI entry point
data/                 # your corpus (not committed)
tests/
```

## Status

Work in progress — built incrementally, one tool at a time.
