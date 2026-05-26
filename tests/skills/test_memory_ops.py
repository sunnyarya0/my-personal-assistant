"""
tests/skills/test_memory_ops.py
================================
Unit tests for the memory skills (agent tools) in skills/memory_ops.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.memory import MemoryEntry, SearchResult
from skills.memory_ops import recall, remember


class TestMemoryOpsSkills:
    @pytest.fixture(autouse=True)
    def mock_store(self):
        self.mock_mem_store = MagicMock()
        with patch("skills.memory_ops.get_memory_store", return_value=self.mock_mem_store):
            yield

    def test_remember_with_tags(self) -> None:
        entry = MemoryEntry(
            id="12345678-abcd-ef01-2345-6789abcdef01",
            content="User prefers dark mode",
            timestamp="2026-05-27T00:00:00Z",
            tags=["ui", "pref"],
        )
        self.mock_mem_store.save.return_value = entry

        res = remember("User prefers dark mode", tags="ui, pref, ")

        assert res == "Remembered (id=12345678): User prefers dark mode"
        self.mock_mem_store.save.assert_called_once_with(
            "User prefers dark mode", tags=["ui", "pref"]
        )

    def test_remember_without_tags(self) -> None:
        entry = MemoryEntry(
            id="87654321-abcd-ef01-2345-6789abcdef01",
            content="User lives in Paris",
            timestamp="2026-05-27T00:00:00Z",
        )
        self.mock_mem_store.save.return_value = entry

        res = remember("User lives in Paris")

        assert res == "Remembered (id=87654321): User lives in Paris"
        self.mock_mem_store.save.assert_called_once_with(
            "User lives in Paris", tags=[]
        )

    def test_recall_no_memories(self) -> None:
        self.mock_mem_store.search.return_value = []

        res = recall("where does the user live?")

        assert res == "No relevant memories found."
        self.mock_mem_store.search.assert_called_once_with(
            "where does the user live?", top_k=5
        )

    def test_recall_with_memories(self) -> None:
        e1 = MemoryEntry(
            id="1",
            content="User likes black coffee",
            timestamp="2026-05-27T10:00:00Z",
            tags=["pref"],
        )
        e2 = MemoryEntry(
            id="2",
            content="User is a software engineer",
            timestamp="2026-05-26T08:00:00Z",
        )
        self.mock_mem_store.search.return_value = [
            SearchResult(entry=e1, score=0.88),
            SearchResult(entry=e2, score=0.72),
        ]

        res = recall("coffee preference", top_k=2)

        expected = (
            "Found 2 relevant memory/memories:\n\n"
            "1. [2026-05-27 [pref]] User likes black coffee  (score: 0.88)\n"
            "2. [2026-05-26] User is a software engineer  (score: 0.72)"
        )
        assert res == expected
        self.mock_mem_store.search.assert_called_once_with(
            "coffee preference", top_k=2
        )
