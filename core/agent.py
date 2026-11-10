"""
core/agent.py
=============
The core ReAct agent reasoning engine.

Implements the Plan → Act → Observe → Refine loop, orchestrating:
  - LLM calls for reasoning and planning
  - Tool execution via the ToolRegistry
  - Safe mode confirmation for destructive tools
  - Verbose step-by-step output to the terminal
  - Session logging to disk

Usage:
    from core.agent import Agent

    agent = Agent()
    result = agent.run("Summarise the files in my Downloads folder")
    print(result.final_answer)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from core.config import settings
from core.llm_client import BaseLLMClient, get_llm_client
from core.memory import MemoryStore, get_memory_store
from core.prompt import (
    AgentAction,
    build_observation_prompt,
    build_system_prompt,
    parse_llm_response,
)
from core.registry import ToolRegistry
from core.registry import registry as _global_registry

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single reasoning + action step in the agent loop."""

    iteration: int
    thought: str
    action: str | None
    action_input: dict[str, Any]
    observation: str
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class AgentResult:
    """The final result returned after the agent loop completes."""

    goal: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    iterations_used: int = 0
    success: bool = True
    error: str | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "final_answer": self.final_answer,
            "iterations_used": self.iterations_used,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "success": self.success,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
        }


def _redact_secrets(data: Any, key_to_redact: str) -> Any:
    """Recursively search and redact occurrences of a sensitive key in a data structure."""
    if not key_to_redact:
        return data
    if isinstance(data, str):
        return data.replace(key_to_redact, "[REDACTED_API_KEY]")
    elif isinstance(data, dict):
        return {k: _redact_secrets(v, key_to_redact) for k, v in data.items()}
    elif isinstance(data, list):
        return [_redact_secrets(item, key_to_redact) for item in data]
    return data


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    Autonomous ReAct agent that uses an LLM to plan and execute tasks.

    The agent runs a loop:
      1. Send the goal (+ history) to the LLM
      2. Parse the LLM response to get a thought + tool call
      3. Execute the tool (with safe mode check if destructive)
      4. Feed the observation back into the next LLM call
      5. Repeat until the LLM provides a final_answer or max_iterations hit
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_store: MemoryStore | None = None,
        verbose: bool | None = None,
        max_iterations: int | None = None,
        safe_mode: bool | None = None,
        agent_name: str | None = None,
        memory_top_k: int = 4,
    ) -> None:
        self.llm = llm_client or get_llm_client()
        self.registry = tool_registry or _global_registry
        self.memory = memory_store or get_memory_store()
        self.verbose = verbose if verbose is not None else settings.verbose
        self.max_iterations = max_iterations or settings.max_iterations
        self.safe_mode = safe_mode if safe_mode is not None else settings.safe_mode
        self.agent_name = agent_name or settings.agent_name
        self.memory_top_k = memory_top_k

        # Ensure log directory exists
        self._log_dir = Path(settings.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        goal: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a goal and return the final result.

        Args:
            goal: The user's natural-language goal or question.
            on_event: Optional callback that receives event notifications.

        Returns:
            AgentResult with the final answer and all intermediate steps.
        """
        start_time = time.perf_counter()
        tools = self.registry.get_all_tools()

        if on_event:
            on_event("agent_start", {"goal": goal})

        # Retrieve relevant memories and inject into system prompt
        try:
            memory_context = self.memory.build_context_block(goal, top_k=self.memory_top_k)
        except Exception as exc:
            logger.warning("Memory retrieval failed (non-fatal): %s", exc)
            memory_context = ""

        base_system_prompt = build_system_prompt(self.agent_name, tools)
        system_prompt = (
            f"{memory_context}\n{base_system_prompt}" if memory_context else base_system_prompt
        )

        if self.verbose:
            self._print_goal_header(goal, tools)

        steps: list[AgentStep] = []
        # Conversation history fed back to the LLM each iteration
        conversation: list[dict[str, str]] = []
        current_prompt = goal

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                console.print(Rule(f"[dim]Iteration {iteration}/{self.max_iterations}[/dim]"))

            # --- Token Budget Warning ---
            approx_tokens = (len(system_prompt) + len(current_prompt)) // 4
            if approx_tokens > settings.llm_max_tokens * 0.8:
                warn_msg = (
                    f"Token Warning: Current prompt uses approx {approx_tokens} tokens, "
                    f"which is nearing the max_tokens limit of {settings.llm_max_tokens}"
                )
                logger.warning(warn_msg)
                if self.verbose:
                    console.print(f"[bold yellow]⚠️  {warn_msg}[/bold yellow]")

            # --- LLM Call ---
            step_start = time.perf_counter()
            try:
                llm_response = self.llm.chat(current_prompt, system_prompt=system_prompt)
                raw_output = llm_response.content
            except Exception as exc:
                error_msg = f"LLM call failed: {exc}"
                logger.error(error_msg)
                if on_event:
                    on_event("agent_error", {"error": error_msg})
                return self._error_result(goal, error_msg, steps, start_time)

            # --- Parse Response ---
            try:
                action = parse_llm_response(raw_output)
            except ValueError as exc:
                # If parsing fails, treat the raw output as a direct answer
                logger.warning("Failed to parse LLM response as JSON: %s", exc)
                action = AgentAction(
                    thought="(Could not parse structured response)",
                    action=None,
                    action_input={},
                    final_answer=raw_output,
                )

            if self.verbose:
                self._print_thought(action.thought)

            if on_event:
                on_event(
                    "agent_step",
                    {
                        "iteration": iteration,
                        "thought": action.thought,
                        "action": action.action,
                        "action_input": action.action_input,
                        "is_done": action.is_done,
                    },
                )

            # --- Done? ---
            if action.is_done:
                elapsed = time.perf_counter() - start_time
                if self.verbose:
                    self._print_final_answer(action.final_answer or "")
                if on_event:
                    on_event(
                        "agent_done",
                        {
                            "final_answer": action.final_answer or "",
                            "iterations_used": iteration,
                        },
                    )
                result = AgentResult(
                    goal=goal,
                    final_answer=action.final_answer or "",
                    steps=steps,
                    total_elapsed_s=elapsed,
                    iterations_used=iteration,
                    success=True,
                )
                self._log_session(result)
                self._auto_save_memory(result)
                return result

            # --- Execute Tool ---
            if action.action is None:
                # LLM gave no action and no answer — treat thought as answer
                elapsed = time.perf_counter() - start_time
                if on_event:
                    on_event(
                        "agent_done",
                        {
                            "final_answer": action.thought,
                            "iterations_used": iteration,
                        },
                    )
                return AgentResult(
                    goal=goal,
                    final_answer=action.thought,
                    steps=steps,
                    total_elapsed_s=elapsed,
                    iterations_used=iteration,
                    success=True,
                )

            if on_event:
                on_event(
                    "tool_start",
                    {
                        "iteration": iteration,
                        "tool": action.action,
                        "input": action.action_input,
                    },
                )

            observation = self._execute_tool(action)
            if on_event:
                on_event(
                    "tool_end",
                    {
                        "iteration": iteration,
                        "tool": action.action,
                        "observation": observation,
                    },
                )
            step_elapsed = (time.perf_counter() - step_start) * 1000

            step = AgentStep(
                iteration=iteration,
                thought=action.thought,
                action=action.action,
                action_input=action.action_input,
                observation=observation,
                elapsed_ms=step_elapsed,
            )
            steps.append(step)

            if self.verbose:
                self._print_tool_call(action.action, action.action_input, observation)

            # --- Build next prompt ---
            conversation.append({"role": "assistant", "content": raw_output})
            conversation.append({"role": "user", "content": f"Tool Result:\n{observation}"})
            current_prompt = build_observation_prompt(conversation, observation)

        # Max iterations exceeded
        elapsed = time.perf_counter() - start_time
        answer = (
            f"I reached the maximum number of steps ({self.max_iterations}) "
            "without completing the task. Here is what I found so far:\n"
            + "\n".join(s.observation for s in steps[-3:])
        )
        result = AgentResult(
            goal=goal,
            final_answer=answer,
            steps=steps,
            total_elapsed_s=elapsed,
            iterations_used=self.max_iterations,
            success=False,
            error="max_iterations_exceeded",
        )
        self._log_session(result)
        return result

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, action: AgentAction) -> str:
        """
        Look up and execute the requested tool, handling safe mode and errors.

        Returns:
            A string observation (the tool's output or an error message).
        """
        tool_schema = self.registry.get(action.action or "")

        if tool_schema is None:
            return (
                f"Error: Tool '{action.action}' is not registered. "
                f"Available tools: {self.registry.names()}"
            )

        # Safe mode confirmation for destructive tools
        if self.safe_mode and tool_schema.destructive:
            if not self._confirm_destructive(action.action or "", action.action_input):
                return "Action cancelled by user (safe mode)."

        try:
            result = tool_schema.function(**action.action_input)
            observation = str(result) if result is not None else "(no output)"
            logger.info(
                "Tool %s called with %s → %s", action.action, action.action_input, observation[:100]
            )
            return observation
        except TypeError as exc:
            return f"Error calling '{action.action}': wrong arguments — {exc}"
        except Exception as exc:
            logger.exception("Tool %s raised an exception", action.action)
            return f"Error calling '{action.action}': {exc}"

    def _confirm_destructive(self, tool_name: str, inputs: dict) -> bool:
        """Prompt the user to confirm a destructive tool call."""
        console.print(
            Panel(
                f"[bold yellow]⚠️  Destructive Action Requested[/bold yellow]\n\n"
                f"Tool: [cyan]{tool_name}[/cyan]\n"
                f"Args: [white]{json.dumps(inputs, indent=2)}[/white]",
                border_style="yellow",
            )
        )
        try:
            confirmed = typer.confirm("Allow this action?", default=False)
            return confirmed
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Logging & Memory
    # ------------------------------------------------------------------

    def _log_session(self, result: AgentResult) -> None:
        """Write a JSON log of the completed agent session to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self._log_dir / f"session_{timestamp}.json"

            data_dict = result.to_dict()
            redacted_dict = _redact_secrets(data_dict, settings.api_key)

            with open(log_file, "w") as f:
                json.dump(redacted_dict, f, indent=2)
            logger.debug("Session logged to %s", log_file)
        except Exception as exc:
            logger.warning("Could not write session log: %s", exc)

    def _auto_save_memory(self, result: AgentResult) -> None:
        """
        Automatically save the user's goal and the agent's final answer
        into long-term memory so the agent remembers what it did.
        """
        if not result.success or not result.final_answer:
            return

        # Don't auto-save if the user's query was just a simple greeting/ping
        if len(result.goal.split()) < 3 and len(result.final_answer.split()) < 10:
            return

        content = f"User Goal: {result.goal}\n" f"Outcome: {result.final_answer}"
        try:
            self.memory.save(content, tags=["auto_save"])
            logger.debug("Auto-saved outcome to long-term memory.")
        except Exception as exc:
            logger.warning("Auto-save to memory failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Rich terminal output helpers
    # ------------------------------------------------------------------

    def _print_goal_header(self, goal: str, tools: list) -> None:
        console.print()
        console.print(
            Panel(
                f"[bold white]{goal}[/bold white]",
                title="[bold cyan]🎯 Goal[/bold cyan]",
                subtitle=(
                    f"[dim]{len(tools)} tools available | "
                    f"max {self.max_iterations} iterations[/dim]"
                ),
                border_style="cyan",
                padding=(0, 2),
            )
        )

    def _print_thought(self, thought: str) -> None:
        console.print(f"\n[bold magenta]💭 Thought:[/bold magenta] [white]{thought}[/white]")

    def _print_tool_call(self, tool_name: str, inputs: dict, observation: str) -> None:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Label", style="bold green", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("🔧 Tool", tool_name)
        table.add_row("📥 Input", json.dumps(inputs))
        table.add_row("📤 Result", observation[:300] + ("..." if len(observation) > 300 else ""))
        console.print(table)

    def _print_final_answer(self, answer: str) -> None:
        console.print()
        console.print(Rule("[bold green]Final Answer[/bold green]"))
        console.print(
            Panel(
                answer,
                title=f"[bold green]{self.agent_name}[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _error_result(
        self,
        goal: str,
        error: str,
        steps: list[AgentStep],
        start_time: float,
    ) -> AgentResult:
        return AgentResult(
            goal=goal,
            final_answer=f"I encountered an error: {error}",
            steps=steps,
            total_elapsed_s=time.perf_counter() - start_time,
            iterations_used=len(steps),
            success=False,
            error=error,
        )


__all__ = ["Agent", "AgentResult", "AgentStep"]
