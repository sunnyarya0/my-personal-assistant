"""
tests/test_registry.py
======================
Unit tests for core/registry.py

Tests cover:
  - Tool registration via function and decorator
  - Schema generation from signatures and docstrings
  - Destructive flag handling
  - Tool lookup and enumeration
  - Auto-discovery of skill files
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.registry import ToolRegistry, ToolSchema, register_tool, registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry() -> ToolRegistry:
    """Return a new empty ToolRegistry for isolated tests."""
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_simple_function(self, fresh_registry: ToolRegistry) -> None:
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        schema = fresh_registry.register(greet)
        assert schema.name == "greet"
        assert "Greet" in schema.description
        assert schema.destructive is False
        assert "greet" in fresh_registry

    def test_register_with_custom_name(self, fresh_registry: ToolRegistry) -> None:
        def fn() -> None:
            """Does nothing."""

        fresh_registry.register(fn, name="my_custom_tool")
        assert "my_custom_tool" in fresh_registry
        assert "fn" not in fresh_registry

    def test_register_with_custom_description(self, fresh_registry: ToolRegistry) -> None:
        def fn() -> None:
            pass

        fresh_registry.register(fn, description="Custom description here")
        schema = fresh_registry.get("fn")
        assert schema is not None
        assert schema.description == "Custom description here"

    def test_register_destructive_flag(self, fresh_registry: ToolRegistry) -> None:
        def nuke() -> None:
            """Deletes everything."""

        fresh_registry.register(nuke, destructive=True)
        schema = fresh_registry.get("nuke")
        assert schema is not None
        assert schema.destructive is True

    def test_register_overwrites_existing(self, fresh_registry: ToolRegistry) -> None:
        def tool() -> None:
            """First version."""

        fresh_registry.register(tool)
        fresh_registry.register(tool, description="Second version")
        assert fresh_registry.get("tool").description == "Second version"


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

class TestSchemaGeneration:
    def test_required_params_detected(self, fresh_registry: ToolRegistry) -> None:
        def move(src: str, dst: str) -> bool:
            """Move a file."""
            return True

        schema = fresh_registry.register(move)
        assert "src" in schema.parameters["required"]
        assert "dst" in schema.parameters["required"]

    def test_optional_params_not_required(self, fresh_registry: ToolRegistry) -> None:
        def search(query: str, limit: int = 10) -> list:
            """Search files."""
            return []

        schema = fresh_registry.register(search)
        assert "query" in schema.parameters["required"]
        assert "limit" not in schema.parameters["required"]
        assert "limit" in schema.parameters["properties"]

    def test_type_mapping(self, fresh_registry: ToolRegistry) -> None:
        def typed_fn(name: str, count: int, ratio: float, flag: bool) -> None:
            """Typed function."""

        schema = fresh_registry.register(typed_fn)
        props = schema.parameters["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"

    def test_no_params_schema(self, fresh_registry: ToolRegistry) -> None:
        def ping() -> str:
            """Ping the server."""
            return "pong"

        schema = fresh_registry.register(ping)
        assert schema.parameters["properties"] == {}
        assert schema.parameters["required"] == []

    def test_to_dict_serialisable(self, fresh_registry: ToolRegistry) -> None:
        import json

        def fn(x: int) -> int:
            """Double a number."""
            return x * 2

        schema = fresh_registry.register(fn)
        d = schema.to_dict()
        # Should be JSON serialisable
        json.dumps(d)
        assert d["name"] == "fn"
        assert "x" in d["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Lookup and enumeration
# ---------------------------------------------------------------------------

class TestLookupAndEnumeration:
    def test_get_existing_tool(self, fresh_registry: ToolRegistry) -> None:
        def my_tool() -> None:
            """A tool."""

        fresh_registry.register(my_tool)
        result = fresh_registry.get("my_tool")
        assert isinstance(result, ToolSchema)

    def test_get_missing_tool_returns_none(self, fresh_registry: ToolRegistry) -> None:
        assert fresh_registry.get("nonexistent") is None

    def test_len(self, fresh_registry: ToolRegistry) -> None:
        assert len(fresh_registry) == 0

        def a() -> None:
            """A."""

        def b() -> None:
            """B."""

        fresh_registry.register(a)
        fresh_registry.register(b)
        assert len(fresh_registry) == 2

    def test_names(self, fresh_registry: ToolRegistry) -> None:
        def alpha() -> None:
            """Alpha."""

        def beta() -> None:
            """Beta."""

        fresh_registry.register(alpha)
        fresh_registry.register(beta)
        assert "alpha" in fresh_registry.names()
        assert "beta" in fresh_registry.names()

    def test_get_all_tools(self, fresh_registry: ToolRegistry) -> None:
        def t1() -> None:
            """T1."""

        def t2() -> None:
            """T2."""

        fresh_registry.register(t1)
        fresh_registry.register(t2)
        tools = fresh_registry.get_all_tools()
        assert len(tools) == 2
        assert all(isinstance(t, ToolSchema) for t in tools)

    def test_get_schema_returns_json_string(self, fresh_registry: ToolRegistry) -> None:
        import json

        def t() -> None:
            """T."""

        fresh_registry.register(t)
        schema_str = fresh_registry.get_schema()
        parsed = json.loads(schema_str)
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "t"


# ---------------------------------------------------------------------------
# @register_tool decorator
# ---------------------------------------------------------------------------

class TestDecorator:
    def test_decorator_no_args(self) -> None:
        local_reg = ToolRegistry()

        @local_reg.register
        def my_fn() -> str:
            """My function."""
            return "ok"

        assert "my_fn" in local_reg

    def test_global_register_tool_decorator(self) -> None:
        """Test the module-level @register_tool on the global registry."""
        initial_count = len(registry)

        @register_tool
        def sprint2_test_tool() -> str:
            """Test tool for sprint 2 registry tests."""
            return "hello"

        assert len(registry) == initial_count + 1
        assert "sprint2_test_tool" in registry

        # Cleanup global registry (best effort)
        registry._tools.pop("sprint2_test_tool", None)

    def test_decorator_with_args(self) -> None:
        local_reg = ToolRegistry()

        def do_register(destructive: bool = False, description: str = ""):
            def decorator(fn):
                local_reg.register(fn, destructive=destructive, description=description or None)
                return fn
            return decorator

        @do_register(destructive=True, description="Wipes data")
        def wipe() -> None:
            pass

        schema = local_reg.get("wipe")
        assert schema is not None
        assert schema.destructive is True
        assert schema.description == "Wipes data"


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

class TestAutoDiscovery:
    def test_discover_skills_from_directory(self, fresh_registry: ToolRegistry) -> None:
        """Test that skill files are auto-imported and tools are registered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "test_skill.py"
            skill_file.write_text(
                "from core.registry import registry\n"
                "def hello_world() -> str:\n"
                "    '''Say hello.'''\n"
                "    return 'hello'\n"
                "registry.register(hello_world)\n"
            )
            count = fresh_registry.discover_skills(Path(tmpdir))
            assert count == 1

    def test_discover_skips_underscore_files(self, fresh_registry: ToolRegistry) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "__init__.py").write_text("")
            (Path(tmpdir) / "_private.py").write_text("")
            count = fresh_registry.discover_skills(Path(tmpdir))
            assert count == 0

    def test_discover_missing_dir_returns_zero(self, fresh_registry: ToolRegistry) -> None:
        count = fresh_registry.discover_skills(Path("/nonexistent/path/skills"))
        assert count == 0
