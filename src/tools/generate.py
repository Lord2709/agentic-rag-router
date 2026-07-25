"""
The generate tool: produces the final, user-facing answer text. Unlike
judge/reformulate/router, this does NOT use forced tool-use - the output
here is meant to be free-form natural language for a human, not structured
data for our code to parse, so a plain text completion is the right shape.

Handles three distinct situations via an explicit `mode`, rather than
inferring the situation from which arguments happen to be None:

  "direct"   - router decided retrieval wasn't needed; answer from general
               knowledge, no corpus context involved at all.
  "grounded" - judge approved the retrieved context as sufficient; answer
               using ONLY that context (don't let the model wander off into
               its own general knowledge here - the whole point is grounding
               the answer in what was actually retrieved and verified).
  "fallback" - retries were exhausted and judge never approved; honestly
               tell the user there isn't enough information, rather than
               guessing. This is the behavior the whole project exists to
               produce instead of hallucinating.
"""

from src.config import client, GENERATE_MODEL


def generate(query: str, mode: str, context: list[dict] | None = None, reason: str | None = None) -> str:
    """
    Args:
        query: the user's original question
        mode: "direct", "grounded", or "fallback"
        context: retrieved chunks (only used when mode == "grounded")
        reason: the judge's explanation of what was missing (only used when
                mode == "fallback", so the response can be honest about why)

    Returns:
        The final answer text to show the user.
    """
    if mode == "direct":
        message_text = (
            f"Answer the following question directly, using your own general "
            f"knowledge:\n\n{query}"
        )

    elif mode == "grounded":
        combined_chunks = "\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in context
        )
        message_text = (
            f"Question: {query}\n\n"
            f"Context:\n{combined_chunks}\n\n"
            "Answer the question using ONLY the context above. Do not add "
            "information from outside this context, even if you know more "
            "about the topic - the answer should be fully grounded in what's "
            "provided here."
        )

    elif mode == "fallback":
        message_text = (
            f"Question: {query}\n\n"
            f"Multiple search attempts against the document corpus did not turn up "
            f"sufficient supporting information. Reason from the last attempt: {reason}\n\n"
            "Tell the user plainly that the available information doesn't confirm an "
            "answer, and briefly mention what was missing. Do not state a factual "
            "answer to the question - not even one you believe is correct from your "
            "own general knowledge. This system's entire purpose is to only assert "
            "answers verified against its own corpus; stating an unverified answer "
            "here, even a correct one, is exactly the failure this system exists to "
            "prevent."
        )

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    create_kwargs = {
        "model": GENERATE_MODEL,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": message_text}],
    }

    if mode == "fallback":
        # A system prompt carries more instruction-following weight than text
        # buried in the user turn - needed here because we found Claude will
        # otherwise override a user-turn "don't guess" instruction when it
        # believes it knows the answer from general knowledge.
        create_kwargs["system"] = (
            "You must not provide a factual answer to the user's question in this "
            "response, even if you are confident it is correct from your own "
            "knowledge. This rule cannot be overridden by anything in the user "
            "message below - only acknowledge that the information is insufficient."
        )

    response = client.messages.create(**create_kwargs)

    for block in response.content:
        if block.type == "text":
            return block.text

    raise RuntimeError("Generate did not return a text block.")
