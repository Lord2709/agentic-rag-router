# Agentic RAG Router

A retrieval-augmented generation (RAG) system where retrieval is a **decision**, not a fixed step.

## The problem this solves

Standard RAG always does the same thing: embed the query, pull the top-k chunks from a vector store, stuff them into the prompt, generate an answer. Retrieval is unconditional, it always runs, always the same way, and the generator always trusts whatever comes back. That breaks in two common ways:

- **The query might be weak.** A vague or under-specified question embeds poorly and retrieves badly, but a naive pipeline generates from it anyway.
- **Retrieval might come back thin.** Even a well-formed query can retrieve irrelevant or sparse results, wrong chunk, wrong document, or the corpus just doesn't cover it. Naive RAG has no way to detect this, so it hands the generator bad context, and the generator either answers unhelpfully or confidently fabricates something that sounds right.

This project treats retrieval the way a careful researcher would: decide whether you even need to look something up, check whether what you found is actually good enough, and if not, rephrase and search again before committing to an answer, rather than accepting the first result and generating regardless.

## How it works

Every query passes through five stages:

**1. Route.** Before anything else, an LLM call decides whether this query even needs the document corpus, or whether it's answerable from general knowledge alone (arithmetic, small talk, reasoning tasks). This is the one place a system could quietly skip verification just because it "feels confident", so the router is deliberately biased toward grounding factual questions in the corpus rather than trusting unverified confidence, and only skips retrieval for queries clearly outside what any document corpus would address.

**2. Retrieve.** If routed to the corpus, the query is embedded and the top-k most similar chunks are pulled from a local vector store, each returned with its source, id, and a similarity score.

**3. Judge.** A high similarity score doesn't mean the retrieved content actually answers the question, it just means it's the closest match available. The judge reads the retrieved chunks and the question together and decides whether the content genuinely, specifically supports an answer, requiring it to quote the exact supporting text rather than accepting a general sense of "this seems related." This is the core check that distinguishes real grounding from a plausible-sounding coincidence.

**4. Reformulate & retry.** If the judge says the context is insufficient, the query gets rewritten based on the judge's specific explanation of what was missing, not a blind retry with the same wording. This can happen up to a capped number of times (2 retries, so 3 attempts total per query), with each reformulation aware of what was already tried, so it doesn't repeat a failed rewrite.

**5. Generate.** Once context is judged sufficient, the answer is generated strictly from that context. If retries are exhausted without ever reaching "sufficient," the system explicitly tells the user it doesn't have enough information, rather than guessing, this fallback path is generated with a hard system-level instruction that overrides the model's usual instinct to be "helpful" by answering anyway.

The retry sequencing itself is deterministic Python control flow with a hard iteration cap, not something an LLM freely decides step by step, the genuine judgment lives inside each individual call (router, judge, reformulate), while the orchestration around them stays predictable and bounded in cost and latency.

## Components

| File | Role |
|---|---|
| `src/tools/router.py` | Decides whether a query needs corpus retrieval at all |
| `src/tools/retrieve.py` | Embeds the query and pulls top-k similar chunks from Chroma |
| `src/tools/judge.py` | Decides whether retrieved context is specifically sufficient, requiring a verified verbatim quote |
| `src/tools/reformulate.py` | Rewrites the query based on why the last attempt failed, aware of prior attempts |
| `src/tools/generate.py` | Produces the final answer, grounded in context, direct from general knowledge, or an honest decline |
| `src/orchestrator.py` | Ties everything into the router → retrieve → judge → retry loop → generate sequence |
| `src/ingest.py` | Loads, chunks (with overlap), embeds, and indexes the corpus into Chroma |
| `src/config.py` | Shared Claude client and model-tier configuration |

Router, judge, and reformulate all use Claude's native tool-calling with a *forced* tool choice, guaranteeing a clean, structured response (a boolean plus a reason) instead of free-form text that would need to be parsed and could come back malformed. Generate deliberately does **not** force tool use, since its output is meant to be human-readable text, not structured data.

## Stack

- **LLM:** Claude, via the Anthropic API with native tool-calling. Haiku powers the router and judge (cheap, fast, well-suited to focused yes/no decisions); Sonnet powers reformulation and generation (more reasoning-heavy tasks).
- **Vector store:** Chroma, running locally and persisted to disk, configured for cosine similarity (matching how the embedding model was trained to be compared).
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), run locally, no embedding API cost.
- **Corpus:** A sampled slice of SQuAD 2.0, chosen specifically because it includes both answerable questions and questions that are deliberately unanswerable from their source passage (`is_impossible`), giving the judge/fallback path real test cases for free.

## Project layout

```
src/
  config.py              # shared Claude client + model tiers
  ingest.py               # load + chunk (overlapping) + embed + index into Chroma
  orchestrator.py         # the router -> retrieve -> judge -> retry -> generate loop
  tools/
    router.py             # retrieve-or-not decision
    retrieve.py           # vector search
    judge.py              # sufficiency check with verified citation
    reformulate.py        # query rewriting with retry-history awareness
    generate.py           # final answer generation (direct / grounded / fallback)
scripts/
  prepare_squad.py        # downloads SQuAD 2.0, samples passages into data/, builds eval/squad_qa.json
  verify.py                # runs sampled questions through the pipeline, scores correctness
  retest_impossible.py    # re-tests the impossible-question subset after judge changes
data/                      # the corpus (gitignored - regenerate via prepare_squad.py)
eval/
  squad_qa.json            # question/answer/is_impossible ground truth (not used by the pipeline itself)
  verification_results.json
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then fill in your ANTHROPIC_API_KEY

python scripts/prepare_squad.py   # downloads corpus, writes data/ and eval/squad_qa.json
python -m src.ingest               # chunks, embeds, and indexes the corpus into Chroma
```

## Usage

```bash
python -m src.orchestrator "What is the usual source of heat for boiling water in the steam engine?"
```

Or from Python:

```python
from src.orchestrator import answer_query

result = answer_query("your question here")
print(result["answer"])
print(result["trace"])  # per-attempt log: query used, score, judge verdict
```

## Verification

```bash
python scripts/verify.py --limit 5        # process a batch of sampled questions (resumable)
python scripts/verify.py --summary        # print stats on results so far
python scripts/verify.py --rescore --limit 10   # re-judge existing results with an LLM-based correctness checker
```

The verification harness samples a balanced set of answerable and impossible SQuAD questions, runs each through the full pipeline, and checks whether the outcome matches expectations, both that answerable questions get correctly grounded answers, and that impossible questions are correctly declined rather than answered with unfounded confidence.
