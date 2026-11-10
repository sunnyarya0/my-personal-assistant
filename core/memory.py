"""
core/memory.py
==============
Two-layer semantic memory system for the Personal AI Agent Assistant.

Layers
------
1. **Short-term** (in-process ring buffer)
   - Stores the last N turns of the *current* session only.
   - Lost when the process exits.
   - Used for immediate conversation context.

2. **Long-term** (ChromaDB vector store, persisted to disk)
   - Survives across sessions.
   - Supports semantic similarity search (RAG).
   - Backed by local sentence-transformers embeddings — no API key needed.

Usage
-----
    from core.memory import get_memory_store

    mem = get_memory_store()
    mem.save("The user prefers Python over JavaScript", tags=["preference"])
    results = mem.search("programming language preference", top_k=3)
    for r in results:
        print(r.content, r.score)
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import settings
from core.embedder import get_embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single memory unit stored in the long-term store."""

    id: str
    content: str
    timestamp: str  # ISO-8601 UTC
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tags=tags or [],
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """A memory entry returned by a semantic search query."""

    entry: MemoryEntry
    score: float  # Cosine similarity [0, 1] — higher = more relevant

    @property
    def content(self) -> str:
        return self.entry.content

    @property
    def id(self) -> str:
        return self.entry.id


# ---------------------------------------------------------------------------
# Short-term memory (in-process ring buffer)
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """
    In-process ring buffer that stores recent conversation turns.
    Cleared when the process exits — useful for maintaining conversation
    context within a single session.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._buffer: deque[str] = deque(maxlen=max_turns)

    def add(self, text: str) -> None:
        """Append a turn to short-term memory."""
        self._buffer.append(text)

    def get_recent(self, n: int | None = None) -> list[str]:
        """Return the most recent N turns (or all if n is None)."""
        items = list(self._buffer)
        if n is not None:
            items = items[-n:]
        return items

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Long-term memory (ChromaDB vector store)
# ---------------------------------------------------------------------------


class LongTermMemory:
    """
    Persistent semantic memory backed by ChromaDB.

    Embeddings are generated locally via sentence-transformers.
    The ChromaDB database is stored on disk at `settings.memory_dir`.
    """

    COLLECTION_NAME = "agent_memory"

    def __init__(self, persist_dir: str | None = None) -> None:
        self._persist_dir = persist_dir or settings.memory_dir
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._embedder = get_embedder()

    def _ensure_connected(self) -> None:
        """Lazy-init ChromaDB client and collection."""
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore

            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB connected at %s (collection=%s, count=%d)",
                self._persist_dir,
                self.COLLECTION_NAME,
                self._collection.count(),
            )
        except ImportError as exc:
            raise ImportError("chromadb is not installed. Run: pip install chromadb") from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Embed and persist a new memory.

        Args:
            content: The text to remember.
            tags: Optional labels for filtering.
            metadata: Optional arbitrary key-value pairs.

        Returns:
            The created MemoryEntry.
        """
        self._ensure_connected()
        entry = MemoryEntry.create(content, tags=tags, metadata=metadata)
        embedding = self._embedder.embed(content)

        chroma_metadata = {
            "timestamp": entry.timestamp,
            "tags": ",".join(entry.tags),
            **{f"meta_{k}": str(v) for k, v in (metadata or {}).items()},
        }

        self._collection.add(
            ids=[entry.id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[chroma_metadata],
        )
        logger.debug("Saved memory %s: %s", entry.id, content[:60])
        return entry

    def forget(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_connected()
        try:
            self._collection.delete(ids=[memory_id])
            logger.debug("Deleted memory %s", memory_id)
            return True
        except Exception as exc:
            logger.warning("Could not delete memory %s: %s", memory_id, exc)
            return False

    def clear(self) -> int:
        """
        Delete ALL memories from the long-term store.

        Returns:
            Number of memories deleted.
        """
        self._ensure_connected()
        count = self._collection.count()
        if count == 0:
            return 0
        # Delete the collection and recreate it
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared %d memories", count)
        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Find the most semantically similar memories to a query.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult ordered by relevance (highest first).
        """
        self._ensure_connected()
        if self._collection.count() == 0:
            return []

        query_embedding = self._embedder.embed(query)
        n_results = min(top_k, self._collection.count())

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        for i, doc in enumerate(raw["documents"][0]):
            meta = raw["metadatas"][0][i]
            chroma_id = raw["ids"][0][i]
            distance = raw["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score [0, 1]
            score = max(0.0, 1.0 - distance / 2.0)

            tags_str = meta.get("tags", "")
            tags = [t for t in tags_str.split(",") if t]

            entry = MemoryEntry(
                id=chroma_id,
                content=doc,
                timestamp=meta.get("timestamp", ""),
                tags=tags,
                metadata={k[5:]: v for k, v in meta.items() if k.startswith("meta_")},
            )
            results.append(SearchResult(entry=entry, score=score))

        return sorted(results, key=lambda r: r.score, reverse=True)

    def list_all(self, limit: int = 50, offset: int = 0) -> list[MemoryEntry]:
        """
        Return all stored memories in reverse chronological order.

        Args:
            limit: Max entries to return.
            offset: Skip this many entries (for pagination).

        Returns:
            List of MemoryEntry objects.
        """
        self._ensure_connected()
        total = self._collection.count()
        if total == 0:
            return []

        raw = self._collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )

        entries: list[MemoryEntry] = []
        for i, doc in enumerate(raw["documents"]):
            meta = raw["metadatas"][i]
            tags_str = meta.get("tags", "")
            tags = [t for t in tags_str.split(",") if t]
            entries.append(
                MemoryEntry(
                    id=raw["ids"][i],
                    content=doc,
                    timestamp=meta.get("timestamp", ""),
                    tags=tags,
                    metadata={k[5:]: v for k, v in meta.items() if k.startswith("meta_")},
                )
            )

        # Sort by timestamp descending (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def count(self) -> int:
        """Return the total number of stored memories."""
        self._ensure_connected()
        return self._collection.count()


# ---------------------------------------------------------------------------
# Unified MemoryStore (facade over both layers)
# ---------------------------------------------------------------------------


class MemoryStore:
    """
    Unified interface combining short-term and long-term memory.

    - Short-term: fast, in-process, session-only.
    - Long-term: persistent, semantic search, cross-session.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        short_term_max: int = 20,
    ) -> None:
        self.short_term = ShortTermMemory(max_turns=short_term_max)
        self.long_term = LongTermMemory(persist_dir=persist_dir)

    def save(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> MemoryEntry:
        """
        Save to short-term (always) and long-term (if persist=True).

        Args:
            content: Text to remember.
            tags: Optional labels.
            metadata: Optional key-value metadata.
            persist: If True, also save to the long-term vector store.

        Returns:
            The MemoryEntry (from long-term store or a stub if not persisted).
        """
        self.short_term.add(content)
        if persist:
            return self.long_term.save(content, tags=tags, metadata=metadata)
        # Return a stub entry for non-persisted saves
        return MemoryEntry.create(content, tags=tags, metadata=metadata)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic similarity search over long-term memories."""
        return self.long_term.search(query, top_k=top_k)

    def forget(self, memory_id: str) -> bool:
        """Delete a specific long-term memory."""
        return self.long_term.forget(memory_id)

    def clear(self, confirm: bool = False) -> int:
        """
        Wipe ALL long-term memories and the short-term buffer.

        Args:
            confirm: Must be True to actually delete (safety guard).

        Returns:
            Number of long-term memories deleted.
        """
        if not confirm:
            logger.warning("clear() called without confirm=True — no action taken.")
            return 0
        self.short_term.clear()
        return self.long_term.clear()

    def list_all(self, limit: int = 50) -> list[MemoryEntry]:
        """List all long-term memories in reverse chronological order."""
        return self.long_term.list_all(limit=limit)

    def count(self) -> int:
        """Total number of long-term memories."""
        return self.long_term.count()

    def build_context_block(self, query: str, top_k: int = 4) -> str:
        """
        Build a formatted memory context block for injection into the
        LLM system prompt.

        Args:
            query: The current user goal (used to find relevant memories).
            top_k: Max memories to inject.

        Returns:
            A formatted string ready to prepend to the system prompt,
            or an empty string if no relevant memories exist.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        lines = ["## Relevant Memories From Previous Sessions"]
        for r in results:
            tag_str = f" [{', '.join(r.entry.tags)}]" if r.entry.tags else ""
            ts = r.entry.timestamp[:10]  # Just the date
            lines.append(f"- [{ts}{tag_str}] {r.content}  (relevance: {r.score:.2f})")
        lines.append("")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    """
    Return the singleton MemoryStore (created on first call).
    Uses the persist_dir from settings.
    """
    return MemoryStore(persist_dir=settings.memory_dir)


__all__ = [
    "MemoryEntry",
    "SearchResult",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryStore",
    "get_memory_store",
]
