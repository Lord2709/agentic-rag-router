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
                "description": "True only if the retrieved context contains enough information to directly and specifically answer the query."
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation. If insufficient, say specifically what's missing or why the context doesn't address the query - this will be used to guide query reformulation."
            }
        },
        "required": ["sufficient", "reason"]
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
        "Does this context contain enough information to directly and "
        "specifically answer the question? Call submit_judgment with your answer."
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
            return block.input

    raise RuntimeError("Judge did not return a tool_use block.")
