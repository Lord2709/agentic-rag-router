"""
The judge tool: reads the query and all retrieved chunks together, and
decides whether the retrieved context actually contains enough information
to answer the question - not just whether it scored highly on similarity.

We just proved (see retrieve() test run) that a high similarity score can
still point at a chunk that doesn't answer the specific question asked.
So this can't be a numeric threshold - it has to genuinely read the content.

Uses Claude with a *forced* tool call to guarantee a structured response
(sufficient: bool, reason: str) rather than parsing free-form text, which
can be malformed or ambiguous.
"""

from src.config import client, JUDGE_MODEL

_JUDGE_TOOL = {
    "name": "submit_judgment",
    "description": "Submit your judgment on whether the retrieved context is sufficient to answer the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sufficient": {
                "type": "boolean",
                "description": (
                    "True only if the retrieved context directly states, or unambiguously and specifically "
                    "entails, the exact answer to the query - reformatting or paraphrasing the question's "
                    "wording is fine (no verbatim match required), but the underlying claim must be clearly "
                    "and specifically supported, not merely plausible. "
                    "False if the context only discusses the same general topic or entity without stating "
                    "the specific fact being asked, or if answering would require guessing, generalizing "
                    "beyond what's stated, or filling a gap the text doesn't actually close. When genuinely "
                    "unsure whether the connection is solid or just plausible, prefer False - a missed "
                    "answer is recoverable via retry, a false 'sufficient' verdict leads directly to a "
                    "confidently wrong answer."
                )
            },
            "supporting_quote": {
                "type": "string",
                "description": (
                    "If sufficient is True, copy the EXACT sentence or phrase from the retrieved context "
                    "(verbatim, character-for-character) that directly supports the answer - not a paraphrase, "
                    "not a summary, an actual quote you can point to in the text above. If sufficient is False, "
                    "leave this empty."
                )
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation. If insufficient, say specifically what's missing or why the context doesn't address the query - this will be used to guide query reformulation."
            }
        },
        "required": ["sufficient", "supporting_quote", "reason"]
    }
}


def judge(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Args:
        query: the user's question (original or reformulated)
        retrieved_chunks: the list of dicts returned by retrieve()
                          (each has "text", "source", "chunk_id", "score")

    Returns:
        {"sufficient": bool, "reason": str}
    """
    combined_chunks = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    message_text = (
        f"Question: {query}\n\n"
        f"Retrieved context:\n{combined_chunks}\n\n"
        "Does this context directly state or unambiguously entail a specific answer "
        "to the question? Paraphrasing/reformatting is fine - do not demand verbatim "
        "wording. But be strict about the underlying claim: if the context only "
        "discusses the same general topic or entity without actually stating the "
        "specific fact asked, or if answering would require generalizing, guessing, "
        "or inferring beyond what the text actually says, mark this insufficient. "
        "If you are genuinely unsure whether the support is solid or just plausible, "
        "prefer insufficient - a wrongly-rejected answer can be recovered by retrying, "
        "but a wrongly-accepted one leads directly to a confidently wrong answer to "
        "the user, which is the one outcome this system must avoid. "
        "If sufficient, you must also copy the exact supporting sentence verbatim into "
        "supporting_quote - if you can't point to a real quote that directly supports "
        "the answer, that's a sign this should actually be insufficient. "
        "Call submit_judgment with your answer."
    )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgment"},
        messages=[{"role": "user", "content": message_text}],
    )

    for block in response.content:
        if block.type == "tool_use":
            verdict = block.input
            return _verify_quote(verdict, combined_chunks)

    raise RuntimeError("Judge did not return a tool_use block.")


def _verify_quote(verdict: dict, combined_chunks: str) -> dict:
    """
    Trust but verify: don't just take the model's word that a supporting quote
    exists - actually check it appears in the retrieved text. Closes the loophole
    where a model claims "sufficient" backed by a quote that sounds plausible but
    isn't actually there (a fabricated citation is just as much a hallucination
    risk as a fabricated answer).
    """
    if not verdict.get("sufficient"):
        return verdict

    quote = (verdict.get("supporting_quote") or "").strip()
    if not quote or quote.lower() not in combined_chunks.lower():
        verdict["sufficient"] = False
        verdict["reason"] = (
            f"Overridden to insufficient: judge claimed sufficient but the supporting "
            f"quote ({quote!r}) does not actually appear verbatim in the retrieved "
            f"context. Original reason: {verdict.get('reason', '')}"
        )

    return verdict
