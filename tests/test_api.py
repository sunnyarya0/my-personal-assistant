"""
tests/test_api.py
=================
Unit tests for the FastAPI backend endpoints in UI/server.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from core.llm_client import LLMResponse
from core.agent import AgentResult, AgentStep
from core.memory import MemoryEntry, SearchResult
from UI.server import app


# Initialize the test client
client = TestClient(app)


class TestApiEndpoints:
    # -----------------------------------------------------------------------
    # Static Assets Delivery
    # -----------------------------------------------------------------------

    def test_serve_index_success(self) -> None:
        with patch("UI.server.Path.exists", return_value=True), \
             patch("UI.server.FileResponse", return_value="mock-html") as mock_response:
            response = client.get("/")
            assert response.status_code == 200
            assert mock_response.called

    def test_serve_index_not_found(self) -> None:
        with patch("UI.server.Path.exists", return_value=False):
            response = client.get("/")
            assert response.status_code == 404
            assert "index.html not found" in response.json()["detail"]

    def test_serve_css_success(self) -> None:
        with patch("UI.server.Path.exists", return_value=True), \
             patch("UI.server.Path.read_text", return_value="body { color: red; }"):
            response = client.get("/index.css")
            assert response.status_code == 200
            assert "text/css" in response.headers["content-type"]
            assert "color: red;" in response.text

    def test_serve_js_success(self) -> None:
        with patch("UI.server.Path.exists", return_value=True), \
             patch("UI.server.Path.read_text", return_value="console.log('hello');"):
            response = client.get("/index.js")
            assert response.status_code == 200
            assert "application/javascript" in response.headers["content-type"]
            assert "hello" in response.text

    # -----------------------------------------------------------------------
    # Memory REST Endpoints
    # -----------------------------------------------------------------------

    def test_list_memories_reverse_chronological(self) -> None:
        mock_mem = MemoryEntry(
            id="1", content="pref", timestamp="2026-05-27T00:00:00Z", tags=["a"]
        )
        mock_store = MagicMock()
        mock_store.list_all.return_value = [mock_mem]

        with patch("UI.server.get_memory_store", return_value=mock_store):
            response = client.get("/api/memory")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "1"
        assert data[0]["content"] == "pref"
        mock_store.list_all.assert_called_once_with(limit=50)

    def test_search_memories_semantic(self) -> None:
        mock_mem = MemoryEntry(
            id="2", content="search match", timestamp="2026-05-27T00:00:00Z", tags=["b"]
        )
        mock_search_res = SearchResult(entry=mock_mem, score=0.8853)
        mock_store = MagicMock()
        mock_store.search.return_value = [mock_search_res]

        with patch("UI.server.get_memory_store", return_value=mock_store):
            response = client.get("/api/memory?query=test")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "2"
        assert data[0]["score"] == 0.8853
        mock_store.search.assert_called_once_with("test", top_k=10)

    def test_delete_memory_success(self) -> None:
        mock_store = MagicMock()
        mock_store.forget.return_value = True

        with patch("UI.server.get_memory_store", return_value=mock_store):
            response = client.delete("/api/memory/some-uuid")

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_store.forget.assert_called_once_with("some-uuid")

    def test_delete_memory_missing_returns_404(self) -> None:
        mock_store = MagicMock()
        mock_store.forget.return_value = False

        with patch("UI.server.get_memory_store", return_value=mock_store):
            response = client.delete("/api/memory/missing-uuid")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    # -----------------------------------------------------------------------
    # Task History Log Endpoints
    # -----------------------------------------------------------------------

    def test_list_tasks_sorts_correctly(self) -> None:
        mock_log_file = MagicMock()
        mock_log_file.stem = "session_20260527_100000"
        mock_log_file.name = "session_20260527_100000.json"
        
        # Mock session contents
        mock_data = {
            "goal": "Test goal",
            "success": True,
            "iterations_used": 3,
            "total_elapsed_s": 1.25,
        }
        mock_log_file.read_text.return_value = json.dumps(mock_data)

        # Mock globbing log directory
        with patch("UI.server.Path.glob", return_value=[mock_log_file]):
            response = client.get("/api/tasks")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "20260527_100000"
        assert data[0]["goal"] == "Test goal"
        assert data[0]["success"] is True

    def test_get_task_details_success(self) -> None:
        mock_data = {"goal": "Details", "success": True, "steps": []}
        
        with patch("UI.server.Path.exists", return_value=True), \
             patch("UI.server.Path.read_text", return_value=json.dumps(mock_data)):
            response = client.get("/api/tasks/20260527_100000")

        assert response.status_code == 200
        assert response.json()["goal"] == "Details"

    def test_get_task_details_not_found(self) -> None:
        with patch("UI.server.Path.exists", return_value=False):
            response = client.get("/api/tasks/missing")

        assert response.status_code == 404

    # -----------------------------------------------------------------------
    # Chat execution REST Endpoint
    # -----------------------------------------------------------------------

    def test_sync_chat_request(self) -> None:
        mock_result = AgentResult(
            goal="Hello",
            final_answer="Response",
            steps=[],
            total_elapsed_s=0.5,
            iterations_used=1,
            success=True,
        )
        
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result

        with patch("UI.server.Agent", return_value=mock_agent), \
             patch("UI.server.registry.discover_skills") as mock_discover:
            response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200
        assert response.json()["final_answer"] == "Response"
        assert response.json()["success"] is True
        mock_discover.assert_called_once()

    # -----------------------------------------------------------------------
    # WebSocket Event Connection
    # -----------------------------------------------------------------------

    def test_websocket_chat_connection(self) -> None:
        mock_result = AgentResult(
            goal="Tell me timezone",
            final_answer="Date time is Tuesday",
            steps=[],
            total_elapsed_s=0.2,
            iterations_used=1,
            success=True,
        )
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result

        with patch("UI.server.Agent", return_value=mock_agent):
            # Using client.websocket_connect simulates Starlette's websocket client
            with client.websocket_connect("/api/ws/chat") as ws:
                ws.send_text('{"goal": "Tell me timezone"}')
                
                # Check for agent completed done message
                data = ws.receive_json()
                assert data["type"] == "agent_done"
                assert data["payload"]["final_answer"] == "Date time is Tuesday"
