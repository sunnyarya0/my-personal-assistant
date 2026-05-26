"""
skills/memory_ops.py
====================
Memory tools that let the agent explicitly remember and recall information.

These are registered as agent tools so the LLM can choose to:
  - Store an important piece of information for later
  - Retrieve relevant memories before answering

Auto-discovered and registered at agent startup.
"""

from __future__ import annotations

from core.memory import get_memory_store
from core.registry import register_tool


@register_tool(description="Save an important piece of information to long-term memory for future sessions.")
def remember(content: str, tags: str = "") -> str:
    """
    Persist a fact, preference, or note to long-term memory.

    Args:
        content: The information to remember.
        tags: Comma-separated labels to categorise this memory (optional).

    Returns:
        Confirmation message with the memory ID.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    mem = get_memory_store()
    entry = mem.save(content, tags=tag_list)
    return f"Remembered (id={entry.id[:8]}): {content[:80]}"


@register_tool(description="Search long-term memory for information relevant to a query.")
def recall(query: str, top_k: int = 5) -> str:
    """
    Retrieve the most relevant memories matching a natural language query.

    Args:
        query: A natural language description of what to look for.
        top_k: Maximum number of memories to return (default 5).

    Returns:
        Formatted list of matching memories with relevance scores,
        or a message indicating no memories were found.
    """
    mem = get_memory_store()
    results = mem.search(query, top_k=top_k)

    if not results:
        return "No relevant memories found."

    lines = [f"Found {len(results)} relevant memory/memories:\n"]
    for i, r in enumerate(results, 1):
        tag_str = f" [{', '.join(r.entry.tags)}]" if r.entry.tags else ""
        ts = r.entry.timestamp[:10]
        lines.append(f"{i}. [{ts}{tag_str}] {r.content}  (score: {r.score:.2f})")

    return "\n".join(lines)
