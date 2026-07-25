"""
The orchestrator: ties router, retrieve, judge, reformulate, and generate
into the actual retry loop this whole project is about.

Deliberately deterministic control flow around genuine LLM decision points
(see the design discussion): router decides retrieve-or-not, judge decides
sufficient-or-not, reformulate decides the next query - but the SEQUENCING
and the hard retry cap are plain Python, not something an LLM freely
decides, so the system has a guaranteed bound on cost/latency no matter
what any individual call decides.
"""

from src.tools.router import route
from src.tools.retrieve import retrieve
from src.tools.judge import judge
from src.tools.reformulate import reformulate_query
from src.tools.generate import generate

MAX_RETRIES = 2  # 2 retries => 3 total retrieval attempts before fallback


def answer_query(query: str) -> dict:
    """
    Args:
        query: the user's question

    Returns:
        {
            "answer": str,
            "used_retrieval": bool,
            "routing_reason": str,
            "attempts": int,
            "sufficient": bool | None,
            "trace": list[dict],
        }
    """
    routing = route(query)

    if not routing["should_retrieve"]:
        answer = generate(query, mode="direct")
        return {
            "answer": answer,
            "used_retrieval": False,
            "routing_reason": routing["reason"],
            "attempts": 0,
            "sufficient": None,
            "trace": [],
        }

    current_query = query
    previous_attempts = []
    trace = []

    for attempt in range(MAX_RETRIES + 1):
        chunks = retrieve(current_query)
        verdict = judge(current_query, chunks)

        trace.append({
            "attempt": attempt + 1,
            "query": current_query,
            "top_score": chunks[0]["score"] if chunks else None,
            "sufficient": verdict["sufficient"],
            "reason": verdict["reason"],
        })

        if verdict["sufficient"]:
            answer = generate(query, mode="grounded", context=chunks)
            return {
                "answer": answer,
                "used_retrieval": True,
                "routing_reason": routing["reason"],
                "attempts": attempt + 1,
                "sufficient": True,
                "trace": trace,
            }

        if attempt < MAX_RETRIES:
            previous_attempts.append(current_query)
            current_query = reformulate_query(query, verdict["reason"], previous_attempts)
        else:
            answer = generate(query, mode="fallback", reason=verdict["reason"])
            return {
                "answer": answer,
                "used_retrieval": True,
                "routing_reason": routing["reason"],
                "attempts": attempt + 1,
                "sufficient": False,
                "trace": trace,
            }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is the usual source of heat for boiling water in the steam engine?"
    result = answer_query(query)
    print(f"Query: {query}\n")
    print(f"Used retrieval: {result['used_retrieval']}")
    print(f"Attempts: {result['attempts']}, Sufficient: {result['sufficient']}")
    print(f"\nAnswer:\n{result['answer']}")
