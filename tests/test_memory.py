"""
tests/test_memory.py
====================
Unit tests for core/memory.py.
Mocks chromadb and sentence_transformers to provide isolated, rapid tests.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
import pytest

from core.memory import (
    MemoryEntry,
    SearchResult,
    ShortTermMemory,
    LongTermMemory,
    MemoryStore,
    get_memory_store,
)


# ---------------------------------------------------------------------------
# MemoryEntry & SearchResult
# ---------------------------------------------------------------------------

class TestMemoryDataModels:
    def test_memory_entry_creation(self) -> None:
        entry = MemoryEntry.create(
            content="fluffy cats are great",
            tags=["animal", "pet"],
            metadata={"priority": "high"},
        )
        assert entry.id is not None
        assert entry.content == "fluffy cats are great"
        assert "animal" in entry.tags
        assert entry.metadata == {"priority": "high"}
        assert entry.timestamp is not None

        # Dict conversion
        d = entry.to_dict()
        assert d["id"] == entry.id
        assert d["content"] == entry.content
        assert d["tags"] == ["animal", "pet"]
        assert d["metadata"] == {"priority": "high"}
        assert d["timestamp"] == entry.timestamp

    def test_search_result_properties(self) -> None:
        entry = MemoryEntry.create("some stored data")
        res = SearchResult(entry=entry, score=0.85)
        assert res.content == "some stored data"
        assert res.id == entry.id
        assert res.score == 0.85


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class TestShortTermMemory:
    def test_short_term_memory_operations(self) -> None:
        mem = ShortTermMemory(max_turns=3)
        assert len(mem) == 0

        mem.add("Turn 1")
        mem.add("Turn 2")
        assert len(mem) == 2
        assert mem.get_recent() == ["Turn 1", "Turn 2"]
        assert mem.get_recent(1) == ["Turn 2"]

        mem.add("Turn 3")
        mem.add("Turn 4")  # Should push out "Turn 1"
        assert len(mem) == 3
        assert mem.get_recent() == ["Turn 2", "Turn 3", "Turn 4"]

        mem.clear()
        assert len(mem) == 0
        assert mem.get_recent() == []


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class TestLongTermMemory:
    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        # Mock embedder
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed.return_value = [0.2] * 384

        # Mock ChromaDB client and collection
        self.mock_collection = MagicMock()
        self.mock_client = MagicMock()
        self.mock_client.get_or_create_collection.return_value = self.mock_collection

        # Mock chromadb module
        self.mock_chromadb_module = MagicMock()
        self.mock_chromadb_module.PersistentClient.return_value = self.mock_client

        with patch("core.memory.get_embedder", return_value=self.mock_embedder), \
             patch.dict("sys.modules", {"chromadb": self.mock_chromadb_module}):
            yield

    def test_lazy_connection(self, tmp_path) -> None:
        fake_path = str(tmp_path / "memory_dir")
        lt = LongTermMemory(persist_dir=fake_path)
        assert lt._client is None
        assert lt._collection is None

        count = lt.count()
        assert lt._client is self.mock_client
        assert lt._collection is self.mock_collection
        self.mock_chromadb_module.PersistentClient.assert_called_once_with(path=fake_path)
        self.mock_client.get_or_create_collection.assert_called_once_with(
            name="agent_memory", metadata={"hnsw:space": "cosine"}
        )

    def test_import_error_raised(self) -> None:
        lt = LongTermMemory()
        with patch.dict("sys.modules", {"chromadb": None}):
            with pytest.raises(ImportError, match="chromadb is not installed"):
                lt.count()

    def test_save_memory(self) -> None:
        self.mock_collection.count.return_value = 5
        lt = LongTermMemory()
        entry = lt.save(
            content="python settings guide",
            tags=["dev", "python"],
            metadata={"project": "assistant"},
        )

        assert entry.content == "python settings guide"
        assert entry.tags == ["dev", "python"]
        assert entry.metadata == {"project": "assistant"}

        # Check call to collection add
        self.mock_collection.add.assert_called_once()
        args, kwargs = self.mock_collection.add.call_args
        assert kwargs["ids"] == [entry.id]
        assert len(kwargs["embeddings"]) == 1
        assert kwargs["documents"] == ["python settings guide"]
        assert kwargs["metadatas"][0]["tags"] == "dev,python"
        assert kwargs["metadatas"][0]["meta_project"] == "assistant"

    def test_forget_memory(self) -> None:
        lt = LongTermMemory()
        assert lt.forget("some-uuid") is True
        self.mock_collection.delete.assert_called_once_with(ids=["some-uuid"])

    def test_forget_failure_returns_false(self) -> None:
        self.mock_collection.delete.side_effect = Exception("ChromaDB error")
        lt = LongTermMemory()
        assert lt.forget("some-uuid") is False

    def test_clear_memory(self) -> None:
        self.mock_collection.count.return_value = 10
        lt = LongTermMemory()
        deleted = lt.clear()

        assert deleted == 10
        self.mock_client.delete_collection.assert_called_once_with("agent_memory")
        assert self.mock_client.get_or_create_collection.call_count == 2  # once in init connect, once in clear

    def test_clear_empty_returns_zero(self) -> None:
        self.mock_collection.count.return_value = 0
        lt = LongTermMemory()
        assert lt.clear() == 0
        self.mock_client.delete_collection.assert_not_called()

    def test_search_empty_store_returns_empty_list(self) -> None:
        self.mock_collection.count.return_value = 0
        lt = LongTermMemory()
        assert lt.search("hello") == []

    def test_search_returns_sorted_results(self) -> None:
        self.mock_collection.count.return_value = 2
        # Mock query results
        self.mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[
                {"timestamp": "2026-05-27T00:00:00Z", "tags": "t1,t2", "meta_priority": "high"},
                {"timestamp": "2026-05-26T00:00:00Z", "tags": "t3", "meta_priority": "low"},
            ]],
            "distances": [[0.2, 0.6]],  # distances: 0.2 similarity = 0.9, 0.6 similarity = 0.7
        }

        lt = LongTermMemory()
        results = lt.search("query text", top_k=2)

        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].content == "doc1"
        assert pytest.approx(results[0].score) == 0.9
        assert results[0].entry.tags == ["t1", "t2"]
        assert results[0].entry.metadata == {"priority": "high"}

        assert results[1].id == "id2"
        assert pytest.approx(results[1].score) == 0.7
        assert results[1].entry.tags == ["t3"]
        assert results[1].entry.metadata == {"priority": "low"}

    def test_list_all_empty(self) -> None:
        self.mock_collection.count.return_value = 0
        lt = LongTermMemory()
        assert lt.list_all() == []

    def test_list_all_paginated_and_sorted(self) -> None:
        self.mock_collection.count.return_value = 3
        # get() return is flattened
        self.mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [
                {"timestamp": "2026-05-26T12:00:00Z", "tags": "a"},
                {"timestamp": "2026-05-27T12:00:00Z", "tags": "b"},
            ],
        }

        lt = LongTermMemory()
        entries = lt.list_all(limit=2, offset=0)

        assert len(entries) == 2
        self.mock_collection.get.assert_called_once_with(
            include=["documents", "metadatas"], limit=2, offset=0
        )
        # Should be sorted with most recent (2026-05-27) first
        assert entries[0].id == "id2"
        assert entries[0].timestamp == "2026-05-27T12:00:00Z"
        assert entries[1].id == "id1"


# ---------------------------------------------------------------------------
# MemoryStore Facade
# ---------------------------------------------------------------------------

class TestMemoryStore:
    @pytest.fixture(autouse=True)
    def mock_components(self) -> None:
        self.mock_lt = MagicMock()
        with patch("core.memory.LongTermMemory", return_value=self.mock_lt):
            yield

    def test_save_with_and_without_persistence(self) -> None:
        store = MemoryStore()

        # Save with persistence
        self.mock_lt.save.return_value = MemoryEntry(
            id="uuid1", content="persistent fact", timestamp="ts", tags=["p"]
        )
        e1 = store.save("persistent fact", tags=["p"], persist=True)
        assert e1.id == "uuid1"
        assert e1.content == "persistent fact"
        assert store.short_term.get_recent() == ["persistent fact"]
        self.mock_lt.save.assert_called_once_with("persistent fact", tags=["p"], metadata=None)

        # Save without persistence
        self.mock_lt.save.reset_mock()
        e2 = store.save("temporary context", persist=False)
        assert e2.content == "temporary context"
        self.mock_lt.save.assert_not_called()
        assert store.short_term.get_recent() == ["persistent fact", "temporary context"]

    def test_delegated_operations(self) -> None:
        store = MemoryStore()

        # search
        store.search("query", top_k=3)
        self.mock_lt.search.assert_called_once_with("query", top_k=3)

        # forget
        store.forget("uuid")
        self.mock_lt.forget.assert_called_once_with("uuid")

        # count
        self.mock_lt.count.return_value = 42
        assert store.count() == 42

        # list_all
        store.list_all(limit=10)
        self.mock_lt.list_all.assert_called_once_with(limit=10)

    def test_clear_with_and_without_confirmation(self) -> None:
        store = MemoryStore()
        store.short_term.add("turn")

        # No confirmation
        deleted = store.clear(confirm=False)
        assert deleted == 0
        assert len(store.short_term) == 1
        self.mock_lt.clear.assert_not_called()

        # With confirmation
        self.mock_lt.clear.return_value = 5
        deleted = store.clear(confirm=True)
        assert deleted == 5
        assert len(store.short_term) == 0
        self.mock_lt.clear.assert_called_once()

    def test_build_context_block_empty(self) -> None:
        store = MemoryStore()
        self.mock_lt.search.return_value = []
        block = store.build_context_block("query")
        assert block == ""

    def test_build_context_block_with_results(self) -> None:
        store = MemoryStore()
        e1 = MemoryEntry(id="1", content="User likes apples", timestamp="2026-05-27T00:00:00Z", tags=["pref"])
        e2 = MemoryEntry(id="2", content="User owns a dog", timestamp="2026-05-26T00:00:00Z")
        self.mock_lt.search.return_value = [
            SearchResult(entry=e1, score=0.9234),
            SearchResult(entry=e2, score=0.7412),
        ]

        block = store.build_context_block("fruit or pets", top_k=2)

        expected = (
            "## Relevant Memories From Previous Sessions\n"
            "- [2026-05-27 [pref]] User likes apples  (relevance: 0.92)\n"
            "- [2026-05-26] User owns a dog  (relevance: 0.74)\n"
        )
        assert block == expected

    def test_singleton_get_memory_store(self) -> None:
        get_memory_store.cache_clear()
        m1 = get_memory_store()
        m2 = get_memory_store()
        assert m1 is m2
