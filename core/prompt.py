"""
core/prompt.py
==============
System prompt builder and LLM response parser for the agent loop.

Responsibilities:
  1. Build a system prompt that tells the LLM how to behave as an agent
     and what tools are available.
  2. Parse raw LLM text output into structured AgentAction objects.

The agent uses a structured JSON output format so the reasoning and
tool calls are machine-readable even from local models that don't
natively support function calling.

Expected LLM output format (one JSON block per turn):
    {
        "thought": "I need to find out what files exist first.",
        "action": "list_directory",
        "action_input": {"path": "/home/user/documents"},
        "final_answer": null
    }

Or when done:
    {
        "thought": "I now have everything I need.",
        "action": null,
        "action_input": null,
        "final_answer": "Here is the summary: ..."
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from core.registry import ToolSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsed action from LLM output
# ---------------------------------------------------------------------------


@dataclass
class AgentAction:
    """Represents one parsed response from the LLM during the ReAct loop."""

    thought: str  # LLM's internal reasoning
    action: str | None  # Tool name to call (None = done)
    action_input: dict[str, Any]  # Arguments for the tool
    final_answer: str | None  # Set when the LLM is done

    @property
    def is_done(self) -> bool:
        """True if the LLM has produced a final answer and wants to stop."""
        return self.final_answer is not None

    def __repr__(self) -> str:
        if self.is_done:
            return f"<AgentAction DONE answer={self.final_answer!r:.40}>"
        return f"<AgentAction action={self.action!r} input={self.action_input}>"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are {agent_name}, an autonomous AI assistant that can accomplish tasks \
by using tools.

## Your Behaviour
- Think step by step before acting.
- Use one tool at a time per response.
- After each tool result, decide if you need another tool or if you have \
enough information to give a final answer.
- Be concise but thorough.
- If a task cannot be completed with the available tools, explain why clearly.

## Available Tools
{tool_schema}

## Response Format
You MUST respond with a single valid JSON object (no markdown, no extra text):

When you need to call a tool:
{{
    "thought": "<your step-by-step reasoning>",
    "action": "<tool_name>",
    "action_input": {{ "<param>": "<value>" }},
    "final_answer": null
}}

When you have a final answer:
{{
    "thought": "<your reasoning for concluding>",
    "action": null,
    "action_input": {{}},
    "final_answer": "<your complete answer to the user>"
}}

## Important Rules
- "action" MUST be one of the tool names listed above, or null.
- "action_input" MUST match the tool's parameter schema exactly.
- Never make up tool names or parameters.
- If you have enough information, set "final_answer" and stop calling tools.
"""

_NO_TOOLS_SYSTEM_PROMPT = """\
You are {agent_name}, a helpful and knowledgeable AI assistant.
Answer the user's question clearly and concisely.
"""


def build_system_prompt(agent_name: str, tools: list[ToolSchema]) -> str:
    """
    Build the agent system prompt, injecting available tool schemas.

    Args:
        agent_name: The name of the agent persona.
        tools: List of registered tools to expose to the LLM.

    Returns:
        Fully rendered system prompt string.
    """
    if not tools:
        return _NO_TOOLS_SYSTEM_PROMPT.format(agent_name=agent_name)

    tool_schema_json = json.dumps(
        [t.to_dict() for t in tools],
        indent=2,
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        tool_schema=tool_schema_json,
    )


def build_observation_prompt(
    history: list[dict[str, str]],
    last_observation: str,
) -> str:
    """
    Build the follow-up user prompt after a tool has been executed,
    feeding the observation back into the loop.

    Args:
        history: List of {"role": ..., "content": ...} turns so far.
        last_observation: The string result of the last tool call.

    Returns:
        A prompt string that feeds the observation back to the LLM.
    """
    return (
        f"Tool Result:\n{last_observation}\n\n"
        "Based on this result, what is your next step? "
        "Respond with the JSON format as instructed."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

# Matches a JSON object (possibly wrapped in ```json ... ``` fences)
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})",
    re.DOTALL,
)


def parse_llm_response(raw: str) -> AgentAction:
    """
    Parse the raw LLM text into an AgentAction.

    Handles:
    - Clean JSON output
    - JSON wrapped in markdown code fences
    - Minor formatting issues from local models

    Args:
        raw: The raw text response from the LLM.

    Returns:
        An AgentAction parsed from the JSON.

    Raises:
        ValueError: If no valid JSON action block could be found.
    """
    raw = raw.strip()

    # Try to find a JSON block
    data: dict[str, Any] | None = None

    # First: try parsing the entire response as JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second: look for JSON block inside the response
    if data is None:
        for match in _JSON_BLOCK_RE.finditer(raw):
            candidate = match.group(1) or match.group(2)
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

    if data is None:
        raise ValueError(
            f"Could not parse a valid JSON action from LLM response.\n"
            f"Raw response:\n{raw[:500]}"
        )

    thought = str(data.get("thought", "")).strip()
    action = data.get("action") or None
    action_input = data.get("action_input") or {}
    final_answer = data.get("final_answer") or None

    # Normalise: if action is the string "null", treat as None
    if isinstance(action, str) and action.lower() in ("null", "none", ""):
        action = None
    if isinstance(final_answer, str) and final_answer.lower() in ("null", "none", ""):
        final_answer = None

    # If no action and no final_answer, treat the thought as the answer
    if action is None and final_answer is None:
        logger.warning("LLM returned neither action nor final_answer; using thought as answer.")
        final_answer = thought

    if not isinstance(action_input, dict):
        action_input = {}

    return AgentAction(
        thought=thought,
        action=action,
        action_input=action_input,
        final_answer=final_answer,
    )


__all__ = [
    "AgentAction",
    "build_system_prompt",
    "build_observation_prompt",
    "parse_llm_response",
]
