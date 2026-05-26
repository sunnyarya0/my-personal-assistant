# Custom Skills Developer Guide

This guide explains how to extend the Personal AI Agent Assistant by writing and registering custom skills.

---

## 1. Skill Mechanics Overview

The assistant auto-discovers skills placed in the `skills/` directory. When the server or CLI boots:
1. It scans all `.py` files inside the `skills/` directory (ignoring files starting with an underscore `_`).
2. It parses functions annotated with the `@register_tool` decorator.
3. It converts function signatures, docstrings, and type hints into JSON schema specs that are injected directly into the LLM system prompt.

---

## 2. Writing a Basic Skill

To write a skill, simply create a new Python file in `skills/` (e.g. `skills/weather.py`) and use the `@register_tool` decorator.

```python
# skills/weather.py
from core.registry import register_tool

@register_tool(description="Get the current weather forecast for a location.")
def get_weather(location: str) -> str:
    """
    Query the weather status.

    Args:
        location: City and country name (e.g. 'Paris, FR').

    Returns:
        A text report of temperature and conditions.
    """
    # Put your API fetch logic here
    return f"The weather in {location} is currently sunny and 22°C."
```

### Critical Rules for Signatures
- **Docstring Structure**: You **must** document arguments inside the docstring since the registry parses them to generate LLM descriptions.
- **Type Hints**: Explicit type hints (e.g. `location: str`) are required for the schema compiler to correctly map inputs. Supported types: `str`, `int`, `float`, `bool`, `list`, `dict`.

---

## 3. Sandboxing & Path Safety

If your custom skill reads or writes to the local file system, you **must** use the `_safe_path` resolver from `skills/file_ops.py` to prevent directory traversal escapes.

```python
from pathlib import Path
from core.registry import register_tool
from skills.file_ops import _safe_path

@register_tool(description="Create a backup copy of a text file inside the workspace.")
def backup_file(source: str, target: str) -> str:
    """
    Safely backup a file.
    """
    try:
        # Enforces workspace path boundary resolver
        safe_source = _safe_path(source)
        safe_target = _safe_path(target)
        
        safe_target.write_text(safe_source.read_text(encoding="utf-8"), encoding="utf-8")
        return f"Successfully backed up {source} to {target}."
    except Exception as exc:
        return f"Backup failed: {exc}"
```

---

## 4. Safety Mode & Destructive Actions

If your skill performs a destructive action (e.g. deleting a resource, sending a network transaction, or modifying a system state), flag it as `destructive=True` inside `@register_tool`.

```python
@register_tool(
    description="Wipe a database file inside the workspace.",
    destructive=True
)
def purge_database(db_name: str) -> str:
    """
    Purge the specified local database file.
    """
    # Purple/Yellow warning panels will trigger in Safe Mode asking for user y/n confirmation.
    ...
```

---

## 5. Testing Custom Skills

All custom skills must be accompanied by unit tests inside `tests/skills/` using mocks to prevent live API dependency in test environments.

```python
# tests/skills/test_weather.py
from unittest.mock import patch
from skills.weather import get_weather

def test_get_weather_success():
    # Test skill logic in isolation
    res = get_weather("Berlin")
    assert "Berlin" in res
    assert "22°C" in res
```
