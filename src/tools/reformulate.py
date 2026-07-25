"""
The reformulate tool: given the original query, the judge's reason for why
the last retrieval attempt was insufficient, and any previously-tried
reformulations this round, propose a new query more likely to retrieve the
missing information.

Passing previous_attempts matters: without it, the model has no memory of
what it already tried this round and could easily propose the same (or a
functionally identical) rewrite twice, burning a retry without genuinely
trying a different angle.

Uses the same forced tool-use pattern as judge.py, for the same reason:
guaranteed clean string output, nothing to parse out of free text.
"""

from src.config import client, REFORMULATE_MODEL

_REFORMULATE_TOOL = {
    "name": "submit_reformulation",
    "description": "Submit a reformulated version of the query designed to retrieve better matching context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reformulated_query": {
                "type": "string",
                "description": "A rewritten version of the question that addresses the gap described in the reason, phrased distinctly enough from any previous attempts to plausibly retrieve different context."
            }
        },
        "required": ["reformulated_query"]
    }
}


def reformulate_query(original_query: str, reason: str, previous_attempts: list[str] | None = None) -> str:
    """
    Args:
        original_query: the question we're ultimately trying to answer
        reason: the judge's explanation of why the last retrieval attempt
                didn't contain enough information
        previous_attempts: reformulated queries already tried this round
                           (empty/None on the first retry)

    Returns:
        A single reformulated query string.
    """
    previous_attempts = previous_attempts or []

    message_text = (
        f"Original question: {original_query}\n\n"
        f"Why the last retrieval attempt failed: {reason}\n\n"
    )

    if previous_attempts:
        tried = "\n".join(f"- {q}" for q in previous_attempts)
        message_text += (
            f"Already tried this round and still failed:\n{tried}\n\n"
            "Propose a genuinely different reformulation - not a minor variation "
            "of the attempts above.\n\n"
        )

    message_text += (
        "Propose a reformulated version of the question that is more likely "
        "to retrieve the specific information needed. Call submit_reformulation."
    )

    response = client.messages.create(
        model=REFORMULATE_MODEL,
        max_tokens=300,
        tools=[_REFORMULATE_TOOL],
        tool_choice={"type": "tool", "name": "submit_reformulation"},
        messages=[{"role": "user", "content": message_text}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input["reformulated_query"]

    raise RuntimeError("Reformulator did not return a tool_use block.")
