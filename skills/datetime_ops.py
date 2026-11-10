"""
skills/datetime_ops.py
======================
Calendar and date/time tracking skill for the Personal AI Agent Assistant.
"""

from __future__ import annotations

import datetime

from core.llm_client import get_llm_client
from core.registry import register_tool


@register_tool(description="Get the current local system date, time, and timezone.")
def get_current_datetime() -> str:
    """
    Query the active system time.

    Returns:
        A text representation of the current datetime, including weekday and timezone offset.
    """
    now = datetime.datetime.now().astimezone()
    weekday = now.strftime("%A")
    return (
        f"Current Datetime: {now.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Weekday: {weekday} | Timezone: {now.tzname()}"
    )


@register_tool(
    description=(
        "Resolve a natural language date expression "
        "(e.g. 'tomorrow', 'next Monday') to YYYY-MM-DD."
    )
)
def parse_date(natural_language: str) -> str:
    """
    Translate natural language date references to absolute YYYY-MM-DD ISO format.

    Args:
        natural_language: Natural language date reference (e.g. 'three days ago', 'next Tuesday').

    Returns:
        The resolved YYYY-MM-DD date or an error message.
    """
    now = datetime.datetime.now()
    nl_lower = natural_language.lower().strip()

    # 1. Simple local matching for common offsets
    if nl_lower in ("today", "now", "current date"):
        return now.strftime("%Y-%m-%d")
    elif nl_lower == "tomorrow":
        return (now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif nl_lower == "yesterday":
        return (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # Match weekdays, e.g. "next monday"
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, w in enumerate(weekdays):
        if w in nl_lower:
            current_day = now.weekday()
            target_day = i
            days_ahead = target_day - current_day
            if "next" in nl_lower or days_ahead <= 0:
                days_ahead += 7
            return (now + datetime.timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # 2. LLM Fallback Date Resolver for complex offsets (e.g. "3 weeks ago")
    try:
        client = get_llm_client()
        system_prompt = (
            f"You are a precise date resolution engine. Today's date is {now.strftime('%Y-%m-%d')} "
            f"({now.strftime('%A')}). Output ONLY the resolved date in YYYY-MM-DD format. "
            f"If the date cannot be resolved or is invalid, output: UNKNOWN"
        )
        response = client.chat(f"Resolve date: '{natural_language}'", system_prompt=system_prompt)
        result = response.content.strip()

        # Simple verification of YYYY-MM-DD shape
        if len(result) == 10 and result[4] == "-" and result[7] == "-":
            return result
        return f"Error: Could not resolve natural language date '{natural_language}'."
    except Exception as exc:
        return f"Error resolving date: {exc}"
