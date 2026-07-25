"""
The router: the one genuinely "routing" decision in this system - does
answering this query well require searching the document corpus, or can
it be confidently answered from general knowledge alone?

Same forced tool-use pattern as judge.py and reformulate.py: guarantees a
clean {"should_retrieve": bool, "reason": str} back, no free-text parsing.
"""

from src.config import client, ROUTER_MODEL

_ROUTER_TOOL = {
    "name": "submit_routing_decision",
    "description": "Decide whether this query requires retrieving from the document corpus, or can be answered directly from general knowledge.",
    "input_schema": {
        "type": "object",
        "properties": {
            "should_retrieve": {
                "type": "boolean",
                "description": (
                    "Default to True whenever the query asks about a specific factual topic that a "
                    "document corpus could plausibly cover (historical events, places, people, technical "
                    "or scientific facts) - ground the answer in a verified source even if you believe you "
                    "already know the answer, since unverified confidence is exactly the failure mode this "
                    "system is designed to avoid. Only use False for queries that are clearly outside what "
                    "any document corpus would address at all: arithmetic/math, greetings and small talk, "
                    "opinions, or requests to reason/summarize/write something with no factual lookup involved."
                )
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation for the routing decision."
            }
        },
        "required": ["should_retrieve", "reason"]
    }
}


def route(query: str) -> dict:
    """
    Args:
        query: the user's question

    Returns:
        {"should_retrieve": bool, "reason": str}
    """
    message_text = (
        f"Question: {query}\n\n"
        "This system has access to a document corpus made up of Wikipedia-style "
        "passages covering assorted historical, scientific, and technical topics "
        "(specific facts about specific entities and events). "
        "Prefer grounding factual questions in that corpus rather than relying on "
        "your own unverified knowledge, even if you believe you already know the "
        "answer - only skip retrieval for queries that are clearly outside what "
        "any document corpus would address (math, small talk, opinions, reasoning "
        "tasks with no factual lookup). "
        "Call submit_routing_decision with your answer."
    )

    response = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=200,
        tools=[_ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "submit_routing_decision"},
        messages=[{"role": "user", "content": message_text}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError("Router did not return a tool_use block.")
