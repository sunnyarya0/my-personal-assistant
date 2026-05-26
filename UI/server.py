"""
UI/server.py
============
FastAPI backend server for the Personal AI Agent Assistant Web Dashboard.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import logging
from pathlib import Path
import time
from typing import Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from core.config import settings
from core.memory import get_memory_store
from core.agent import Agent
from core.registry import registry

# Configure logging
logger = logging.getLogger(__name__)

# In-memory rate limiting state
rate_limit_records: dict[str, list[float]] = defaultdict(list)


async def rate_limiter(request: Request) -> None:
    """Enforce rate limits per client IP for /api routes."""
    if not request.url.path.startswith("/api"):
        return
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    timestamps = rate_limit_records[client_ip]
    rate_limit_records[client_ip] = [t for t in timestamps if now - t < 60]
    
    if len(rate_limit_records[client_ip]) >= settings.rate_limit_rpm:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests: Rate limit exceeded."
        )
    rate_limit_records[client_ip].append(now)


app = FastAPI(
    title="🤖 Personal AI Agent Assistant Dashboard",
    description="Control panel and monitoring UI for your autonomous agent.",
    version="0.1.0",
)

# Enable CORS for frontend asset delivery
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required storage directories exist
settings.ensure_directories()

# Static directories path
UI_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Static assets delivery
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index() -> FileResponse:
    """Serve the primary HTML dashboard."""
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_path)


@app.get("/index.css")
async def serve_css() -> Response:
    """Serve the CSS style sheet with proper MIME type."""
    css_path = UI_DIR / "index.css"
    if not css_path.exists():
        return Response(content="/* CSS not found */", media_type="text/css")
    return Response(content=css_path.read_text(encoding="utf-8"), media_type="text/css")


@app.get("/index.js")
async def serve_js() -> Response:
    """Serve the frontend JS script with proper MIME type."""
    js_path = UI_DIR / "index.js"
    if not js_path.exists():
        return Response(content="// JS not found", media_type="application/javascript")
    return Response(content=js_path.read_text(encoding="utf-8"), media_type="application/javascript")


# ---------------------------------------------------------------------------
# Memory REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/memory", dependencies=[Depends(rate_limiter)])
async def list_memories(query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """
    List or search memories.
    If a query parameter is provided, performs a semantic search.
    Otherwise, lists memories in reverse-chronological order.
    """
    try:
        mem = get_memory_store()
        if query:
            results = mem.search(query, top_k=min(limit, 10))
            return [
                {
                    "id": r.entry.id,
                    "content": r.entry.content,
                    "timestamp": r.entry.timestamp,
                    "tags": r.entry.tags,
                    "metadata": r.entry.metadata,
                    "score": round(r.score, 4),
                }
                for r in results
            ]
        else:
            entries = mem.list_all(limit=limit)
            return [e.to_dict() for e in entries]
    except Exception as exc:
        logger.exception("Failed to query memories")
        raise HTTPException(status_code=500, detail=f"Failed to query memories: {exc}")


@app.delete("/api/memory/{memory_id}", dependencies=[Depends(rate_limiter)])
async def delete_memory(memory_id: str) -> dict[str, str]:
    """Delete a specific memory by ID."""
    try:
        mem = get_memory_store()
        success = mem.forget(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory entry not found.")
        return {"status": "success", "message": f"Memory {memory_id} deleted."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {exc}")


# ---------------------------------------------------------------------------
# Task logs REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/tasks", dependencies=[Depends(rate_limiter)])
async def list_tasks() -> list[dict[str, Any]]:
    """List past tasks and runs stored in the log directory."""
    try:
        log_dir = Path(settings.log_dir)
        sessions = []
        for file in log_dir.glob("session_*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                # Extract simple summary structure
                sessions.append({
                    "id": file.stem.replace("session_", ""),
                    "goal": data.get("goal", ""),
                    "success": data.get("success", False),
                    "timestamp": data.get("timestamp", datetime_from_filename(file.name)),
                    "iterations": data.get("iterations_used", 0),
                    "elapsed_s": round(data.get("total_elapsed_s", 0.0), 2),
                })
            except Exception:
                continue

        # Sort with newest first
        sessions.sort(key=lambda s: s["id"], reverse=True)
        return sessions
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load task logs: {exc}")


@app.get("/api/tasks/{task_id}", dependencies=[Depends(rate_limiter)])
async def get_task_details(task_id: str) -> dict[str, Any]:
    """Fetch complete step-by-step logs for a specific task."""
    import re
    if not re.match(r"^[a-zA-Z0-9_]+$", task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format.")

    try:
        log_dir = Path(settings.log_dir).resolve()
        log_file = (log_dir / f"session_{task_id}.json").resolve()
        if not log_file.is_relative_to(log_dir) or not log_file.exists():
            raise HTTPException(status_code=404, detail="Task log file not found.")
        return json.loads(log_file.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read task log: {exc}")


def datetime_from_filename(filename: str) -> str:
    """Helper to parse datetime from log filename session_YYYYMMDD_HHMMSS.json."""
    try:
        parts = filename.replace("session_", "").replace(".json", "").split("_")
        if len(parts) == 2:
            d, t = parts[0], parts[1]
            return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    except Exception:
        pass
    return "Unknown Date"


# ---------------------------------------------------------------------------
# Non-streaming Chat Request
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    max_iterations: int | None = None


@app.post("/api/chat", dependencies=[Depends(rate_limiter)])
async def chat_request(req: ChatRequest) -> dict[str, Any]:
    """Execute the agent synchronously and return the final response."""
    try:
        # Discover skills in case new ones were added
        registry.discover_skills()
        
        agent = Agent(max_iterations=req.max_iterations)
        result = await run_in_threadpool(agent.run, req.message)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")


# ---------------------------------------------------------------------------
# WebSocket Bidirectional Event Streaming
# ---------------------------------------------------------------------------

@app.websocket("/api/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """ WebSocket chat endpoint streaming intermediate agent thoughts live. """
    await websocket.accept()
    logger.info("WebSocket connection established.")

    loop = asyncio.get_running_loop()

    # Callback mapping sync Agent events to async WebSocket send
    def send_socket_event(event_type: str, data: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": event_type, "payload": data}),
            loop
        )

    try:
        while True:
            # Await goal message from the client
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                goal = payload.get("goal", "")
            except Exception:
                goal = data

            if not goal or not goal.strip():
                await websocket.send_json({
                    "type": "agent_error",
                    "payload": {"error": "Empty prompt received."}
                })
                continue

            # Load skills
            registry.discover_skills()
            agent = Agent()

            # Execute the agent on a worker thread pool, streaming progress synchronously
            try:
                # Runs Agent.run in threadpool while events marshal back safely
                result = await run_in_threadpool(agent.run, goal, send_socket_event)
                await websocket.send_json({
                    "type": "agent_done",
                    "payload": {
                        "final_answer": result.final_answer,
                        "success": result.success,
                        "iterations": result.iterations_used,
                    }
                })
            except Exception as exc:
                logger.error("Websocket agent execution failed: %s", exc)
                await websocket.send_json({
                    "type": "agent_error",
                    "payload": {"error": f"Execution error: {exc}"}
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as exc:
        logger.error("WebSocket exception encountered: %s", exc)
