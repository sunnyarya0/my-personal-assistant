"""
core/registry.py
================
Centralized registry for all agent tools and skills.

Tools are registered via the @register_tool decorator and are
auto-discovered by importing all .py files under skills/.

Usage:
    from core.registry import register_tool, registry

    @register_tool(description="Adds two numbers", destructive=False)
    def add(a: int, b: int) -> int:
        \"\"\"Add two integers and return their sum.\"\"\"
        return a + b

    all_tools = registry.get_all_tools()
    schema    = registry.get_schema()        # For injecting into LLM prompt
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """Metadata and schema for a single registered tool."""
    name: str
    description: str
    destructive: bool
    function: Callable
    parameters: dict[str, Any]   # JSON-schema-style parameter spec

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation for prompt injection."""
        return {
            "name": self.name,
            "description": self.description,
            "destructive": self.destructive,
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r} destructive={self.destructive}>"


# ---------------------------------------------------------------------------
# Type → JSON Schema helpers
# ---------------------------------------------------------------------------

_PY_TYPE_TO_JSON: dict[Any, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_type(annotation: Any) -> str:
    """Convert a Python type annotation to its JSON schema equivalent."""
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _PY_TYPE_TO_JSON.get(annotation, "string")


def _build_parameter_schema(func: Callable) -> dict[str, Any]:
    """
    Introspect a function's signature and build a JSON-schema-style
    parameter spec that can be injected into the LLM prompt.
    """
    import typing

    sig = inspect.signature(func)

    # get_type_hints is the most reliable way to resolve annotations
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = getattr(func, "__annotations__", {})

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, str)
        json_type = _python_type_to_json_type(annotation)

        prop: dict[str, Any] = {"type": json_type}

        # Extract per-param description from docstring (Google style)
        doc = inspect.getdoc(func) or ""
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{param_name}:") or stripped.startswith(f"{param_name} ("):
                desc = stripped.split(":", 1)[-1].strip()
                if desc:
                    prop["description"] = desc
                break

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Singleton-style registry that stores all registered tools.

    Tools self-register via the @register_tool decorator.
    Use registry.get_all_tools() to enumerate them.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        destructive: bool = False,
    ) -> ToolSchema:
        """
        Register a callable as an agent tool.

        Args:
            func: The Python function to register.
            name: Override the tool name (defaults to func.__name__).
            description: Override the description (defaults to first line of docstring).
            destructive: If True, safe mode will prompt for confirmation.

        Returns:
            The created ToolSchema.
        """
        tool_name = name or func.__name__

        # Derive description from first line of docstring if not provided
        if description is None:
            doc = inspect.getdoc(func) or ""
            description = doc.splitlines()[0] if doc else f"Tool: {tool_name}"

        schema = ToolSchema(
            name=tool_name,
            description=description,
            destructive=destructive,
            function=func,
            parameters=_build_parameter_schema(func),
        )
        self._tools[tool_name] = schema
        logger.debug("Registered tool: %s (destructive=%s)", tool_name, destructive)
        return schema

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSchema | None:
        """Return a ToolSchema by name, or None if not found."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[ToolSchema]:
        """Return all registered tools in registration order."""
        return list(self._tools.values())

    def get_schema(self) -> str:
        """
        Return a JSON string listing all tools with their schemas.
        Used for injecting into LLM system prompts.
        """
        return json.dumps(
            [t.to_dict() for t in self._tools.values()],
            indent=2,
        )

    def names(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.names()}>"

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def discover_skills(self, skills_dir: Path | None = None) -> int:
        """
        Auto-import all .py files in the skills/ directory so their
        @register_tool decorators run and self-register.

        Args:
            skills_dir: Path to the skills directory. Defaults to
                        <project_root>/skills/

        Returns:
            Number of skill files imported.
        """
        if skills_dir is None:
            skills_dir = Path(__file__).parent.parent / "skills"

        if not skills_dir.exists():
            logger.warning("Skills directory not found: %s", skills_dir)
            return 0

        count = 0
        for skill_file in sorted(skills_dir.glob("*.py")):
            if skill_file.name.startswith("_"):
                continue
            module_name = f"skills.{skill_file.stem}"
            if module_name in sys.modules:
                continue  # Already imported
            try:
                spec = importlib.util.spec_from_file_location(module_name, skill_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)  # type: ignore
                    count += 1
                    logger.debug("Discovered skill file: %s", skill_file.name)
            except Exception as exc:
                logger.warning("Failed to import skill %s: %s", skill_file.name, exc)

        return count


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = ToolRegistry()


# ---------------------------------------------------------------------------
# @register_tool decorator
# ---------------------------------------------------------------------------

def register_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
) -> Callable:
    """
    Decorator to register a function as an agent tool.

    Can be used with or without arguments:

        @register_tool
        def my_tool(x: int) -> str: ...

        @register_tool(description="Override desc", destructive=True)
        def delete_file(path: str) -> bool: ...

    Args:
        func: The function to register (when used without parentheses).
        name: Override the tool name.
        description: Override the description.
        destructive: Mark this tool as destructive (triggers safe mode).
    """
    def decorator(fn: Callable) -> Callable:
        registry.register(fn, name=name, description=description, destructive=destructive)
        return fn

    if func is not None:
        # Used as @register_tool (no parentheses)
        return decorator(func)

    # Used as @register_tool(...) with arguments
    return decorator


__all__ = [
    "ToolRegistry",
    "ToolSchema",
    "registry",
    "register_tool",
]
