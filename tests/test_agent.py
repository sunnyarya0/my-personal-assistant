"""
tests/test_agent.py
===================
Unit tests for core/agent.py and core/prompt.py

All tests use mocked LLM responses — no live Ollama calls are made.
Tests verify the full ReAct loop behaviour including:
  - Single-step completion (LLM gives final_answer immediately)
  - Multi-step tool use and observation feeding
  - Max iterations guard
  - Tool not found handling
  - Safe mode blocking
  - Parse failure fallback
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.agent import Agent, AgentResult
from core.prompt import AgentAction, build_system_prompt, parse_llm_response
from core.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_llm_response(
    thought: str = "I know the answer.",
    action: str | None = None,
    action_input: dict | None = None,
    final_answer: str | None = "42",
) -> str:
    """Build a JSON string the LLM would return."""
    return json.dumps(
        {
            "thought": thought,
            "action": action,
            "action_input": action_input or {},
            "final_answer": final_answer,
        }
    )


def make_tool_call_response(tool: str, inputs: dict, thought: str = "Let me use a tool.") -> str:
    """Build a JSON response that calls a tool."""
    return json.dumps(
        {
            "thought": thought,
            "action": tool,
            "action_input": inputs,
            "final_answer": None,
        }
    )


def make_agent_with_mock_llm(
    responses: list[str],
    registry: ToolRegistry | None = None,
    verbose: bool = False,
    max_iterations: int = 5,
    safe_mode: bool = False,
) -> Agent:
    """
    Create an Agent with a mock LLM that returns responses in sequence.
    """
    mock_llm = MagicMock()
    mock_responses = [MagicMock(content=r) for r in responses]
    mock_llm.chat.side_effect = mock_responses

    reg = registry or ToolRegistry()
    return Agent(
        llm_client=mock_llm,
        tool_registry=reg,
        verbose=verbose,
        max_iterations=max_iterations,
        safe_mode=safe_mode,
        agent_name="TestBot",
    )


# ---------------------------------------------------------------------------
# parse_llm_response tests
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_parse_clean_final_answer(self) -> None:
        raw = make_llm_response(thought="Done.", final_answer="The answer is 7.")
        action = parse_llm_response(raw)
        assert action.is_done is True
        assert action.final_answer == "The answer is 7."
        assert action.action is None

    def test_parse_tool_call(self) -> None:
        raw = make_tool_call_response("list_files", {"path": "/tmp"})
        action = parse_llm_response(raw)
        assert action.is_done is False
        assert action.action == "list_files"
        assert action.action_input == {"path": "/tmp"}

    def test_parse_json_wrapped_in_markdown(self) -> None:
        raw = '```json\n{"thought":"ok","action":null,"action_input":{},"final_answer":"done"}\n```'
        action = parse_llm_response(raw)
        assert action.is_done is True
        assert action.final_answer == "done"

    def test_parse_null_action_as_none(self) -> None:
        raw = '{"thought":"t","action":"null","action_input":{},"final_answer":null}'
        action = parse_llm_response(raw)
        assert action.action is None

    def test_parse_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_response("This is just plain text with no JSON at all.")

    def test_parse_thought_used_as_answer_when_no_action_or_final(self) -> None:
        raw = '{"thought":"I concluded.","action":null,"action_input":{},"final_answer":null}'
        action = parse_llm_response(raw)
        # When no action and no final_answer, thought becomes the answer
        assert action.final_answer == "I concluded."


# ---------------------------------------------------------------------------
# AgentAction tests
# ---------------------------------------------------------------------------


class TestAgentAction:
    def test_is_done_true_when_final_answer_set(self) -> None:
        a = AgentAction(thought="t", action=None, action_input={}, final_answer="done")
        assert a.is_done is True

    def test_is_done_false_when_no_final_answer(self) -> None:
        a = AgentAction(thought="t", action="tool", action_input={}, final_answer=None)
        assert a.is_done is False


# ---------------------------------------------------------------------------
# Agent single-step (no tools needed)
# ---------------------------------------------------------------------------


class TestAgentSingleStep:
    def test_returns_final_answer_immediately(self) -> None:
        agent = make_agent_with_mock_llm(
            responses=[make_llm_response(final_answer="Paris is the capital of France.")]
        )
        result = agent.run("What is the capital of France?")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "Paris" in result.final_answer
        assert result.iterations_used == 1
        assert result.step_count == 0  # No tool calls

    def test_result_contains_goal(self) -> None:
        goal = "Tell me a joke."
        agent = make_agent_with_mock_llm(
            responses=[make_llm_response(final_answer="Why did the chicken...")]
        )
        result = agent.run(goal)
        assert result.goal == goal


# ---------------------------------------------------------------------------
# Agent multi-step (with tools)
# ---------------------------------------------------------------------------


class TestAgentMultiStep:
    def test_calls_tool_then_gives_answer(self) -> None:
        # Set up a registry with a simple tool
        reg = ToolRegistry()

        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Sunny, 22°C in {city}"

        reg.register(get_weather)

        # LLM first calls the tool, then gives final answer
        responses = [
            make_tool_call_response("get_weather", {"city": "Paris"}),
            make_llm_response(final_answer="The weather in Paris is Sunny, 22°C."),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg)
        result = agent.run("What is the weather in Paris?")

        assert result.success is True
        assert result.step_count == 1
        assert result.steps[0].action == "get_weather"
        assert "Paris" in result.steps[0].observation

    def test_multiple_tool_calls(self) -> None:
        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        reg.register(add)
        reg.register(multiply)

        responses = [
            make_tool_call_response("add", {"a": 3, "b": 4}),
            make_tool_call_response("multiply", {"a": 7, "b": 2}),
            make_llm_response(final_answer="3 + 4 = 7, and 7 × 2 = 14."),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg)
        result = agent.run("What is (3+4)*2?")

        assert result.success is True
        assert result.step_count == 2
        assert result.steps[0].action == "add"
        assert result.steps[1].action == "multiply"

    def test_observation_contains_tool_output(self) -> None:
        reg = ToolRegistry()

        def echo(msg: str) -> str:
            """Echo a message back."""
            return f"ECHO: {msg}"

        reg.register(echo)

        responses = [
            make_tool_call_response("echo", {"msg": "hello"}),
            make_llm_response(final_answer="The echo said: ECHO: hello"),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg)
        result = agent.run("Echo hello.")

        assert "ECHO: hello" in result.steps[0].observation


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestAgentErrorHandling:
    def test_unknown_tool_returns_error_observation(self) -> None:
        reg = ToolRegistry()  # Empty — no tools

        responses = [
            make_tool_call_response("nonexistent_tool", {}),
            make_llm_response(final_answer="I could not use that tool."),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg)
        result = agent.run("Do something.")

        assert "not registered" in result.steps[0].observation

    def test_max_iterations_exceeded(self) -> None:
        reg = ToolRegistry()

        def noop() -> str:
            """Does nothing."""
            return "nothing"

        reg.register(noop)

        # Always call the tool, never give final answer
        responses = [make_tool_call_response("noop", {})] * 10
        agent = make_agent_with_mock_llm(responses, registry=reg, max_iterations=3)
        result = agent.run("Keep going forever.")

        assert result.success is False
        assert result.error == "max_iterations_exceeded"
        assert result.iterations_used == 3

    def test_llm_failure_returns_error_result(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("Connection refused")

        agent = Agent(
            llm_client=mock_llm,
            tool_registry=ToolRegistry(),
            verbose=False,
            max_iterations=3,
            safe_mode=False,
        )
        result = agent.run("Do something.")

        assert result.success is False
        assert "Connection refused" in result.error or "Connection refused" in result.final_answer

    def test_parse_failure_uses_raw_as_answer(self) -> None:
        """If LLM returns non-JSON, the raw text becomes the final answer."""
        agent = make_agent_with_mock_llm(responses=["The answer is simply 42, my friend."])
        result = agent.run("What is the answer to life?")

        assert result.success is True
        assert "42" in result.final_answer

    def test_tool_wrong_args_returns_error_observation(self) -> None:
        reg = ToolRegistry()

        def strict_fn(required_arg: str) -> str:
            """Needs an argument."""
            return required_arg

        reg.register(strict_fn)

        responses = [
            # Call with wrong argument name
            make_tool_call_response("strict_fn", {"wrong_arg": "value"}),
            make_llm_response(final_answer="There was an error."),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg)
        result = agent.run("Run strict_fn.")

        assert "Error" in result.steps[0].observation


# ---------------------------------------------------------------------------
# Safe mode
# ---------------------------------------------------------------------------


class TestSafeMode:
    def test_safe_mode_blocks_destructive_tool(self) -> None:
        reg = ToolRegistry()

        def delete_all() -> str:
            """Delete all files."""
            return "deleted"

        reg.register(delete_all, destructive=True)

        responses = [
            make_tool_call_response("delete_all", {}),
            make_llm_response(final_answer="Action was cancelled."),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg, safe_mode=True)

        # Patch typer.confirm to return False (user rejects)
        with patch("core.agent.typer.confirm", return_value=False):
            result = agent.run("Delete all files.")

        assert "cancelled" in result.steps[0].observation.lower()

    def test_safe_mode_allows_when_confirmed(self) -> None:
        reg = ToolRegistry()

        def delete_file(path: str) -> str:
            """Delete a file."""
            return f"Deleted {path}"

        reg.register(delete_file, destructive=True)

        responses = [
            make_tool_call_response("delete_file", {"path": "/tmp/test.txt"}),
            make_llm_response(final_answer="File deleted."),
        ]
        agent = make_agent_with_mock_llm(responses, registry=reg, safe_mode=True)

        # Patch typer.confirm to return True (user confirms)
        with patch("core.agent.typer.confirm", return_value=True):
            result = agent.run("Delete /tmp/test.txt")

        assert "Deleted" in result.steps[0].observation


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_prompt_includes_tool_names(self) -> None:
        reg = ToolRegistry()

        def search_web(query: str) -> str:
            """Search the web."""
            return ""

        def read_file(path: str) -> str:
            """Read a file."""
            return ""

        reg.register(search_web)
        reg.register(read_file)

        prompt = build_system_prompt("Jarvis", reg.get_all_tools())
        assert "search_web" in prompt
        assert "read_file" in prompt

    def test_prompt_with_no_tools(self) -> None:
        prompt = build_system_prompt("Jarvis", [])
        assert "Jarvis" in prompt
        # Should use simpler prompt when no tools
        assert "Available Tools" not in prompt

    def test_prompt_includes_agent_name(self) -> None:
        prompt = build_system_prompt("Alfred", [])
        assert "Alfred" in prompt
