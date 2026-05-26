# Changelog

All notable changes to the Personal AI Agent Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-05-27
### Added
- **API Rate Limiting**: Built-in client-IP-based endpoint rate limiting to prevent FastAPI backend overload.
- **Scraper Allowed Domains**: Added `allowed_domains` configuration in `settings.yaml` to restrict web scraping and summarization actions.
- **Log Protection**: Automated recursive search and redaction of `settings.api_key` secrets inside saved task session logs.
- **Command Sanitization**: Strict blocklist verification (`sudo`, recursive outside-workspace path removals, elevation commands) inside the `run_command` terminal skill.
- **Token warnings**: Character/token budget estimator that provides real-time terminal and console warnings when prompt buffers exceed 80% of `settings.llm_max_tokens`.
- **API Exponential Backoff Retry**: Automatic transient error and rate-limit backoff logic with random jitter wrapping all base LLM client requests.
- **CI Pipeline**: Pre-configured GitHub Actions CI YAML workflow executing Black, Ruff, and Pytest across Python 3.10, 3.11, and 3.12.
- **System Documentation**: Expanded user manuals, architecture blueprints, custom skill guidelines, and complete configuration mappings.

---

## [0.5.0] - 2026-05-26
### Added
- **FastAPI Backend Server (`UI/server.py`)**: Asynchronous endpoint infrastructure enabling:
  - REST routes for memory browsing, searching, and deleting.
  - REST routes for listing and reading historical task logs.
  - Bidirectional WebSocket streaming route for real-time thought transmission.
- **Agent Event Hooks**: ReAct agent loops now support synchronous-to-async event marshalling using thread pool callbacks (`agent_start`, `agent_step`, `tool_start`, `tool_end`, `agent_done`, `agent_error`).
- **Serve CLI Command**: Wired `python main.py serve` to automatically launch the FastAPI Uvicorn engine and launch default system browsers.
- **Glassmorphic Web Dashboard**: Highly styled HTML/JS/CSS frontend displaying a multi-column Galactic Dark workspace:
  - Scrolling Live Reasoning Console rendering intermediate plan plans, thoughts, and observations.
  - Sidebars for paginated, searchable, and deletable semantic memories.
  - Clickable past task history listings.
- **REST & WS Endpoint Test Suite**: FastAPI REST/WS mockup validation suite yielding passing client coverage.

---

## [0.4.0] - 2026-05-25
### Added
- **File Operations Skill**: Strictly sandboxed local file tools preventing directory traversal escapes (`_safe_path`).
- **Web Search Skill**: DuckDuckGo Lite parsing scrapper alongside BeautifulSoup webpage text extractors and LLM summarization.
- **System Diagnostics Skill**: Native `run_command` execution together with platform memory/CPU diagnostics and process enumeration.
- **DateTime Skill**: Contextual clock query tool with natural language date resolver fallback.

---

## [0.3.0] - 2026-05-24
### Added
- **Sentence-Transformers Embeddings**: Offline-first vector embeds using `all-MiniLM-L6-v2`.
- **ChromaDB Vector Store**: Cross-session long-term semantic memory storage.
- **Context Injection**: Automated memory retrieval and system prompt enrichment.
- **Auto-Save Memory**: Automatic extraction and storage of successful query outcomes.

---

## [0.2.0] - 2026-05-23
### Added
- **ReAct Execution Engine**: Plan → Act → Observe reasoning loops.
- **Extensible `@register_tool` Decorator**: Dynamic tool schema parser.
- **Auto-Discovery**: Automatic package skill imports.
- **Interactive Safe Mode**: Confirmation queries for destructive tools.

---

## [0.1.0] - 2026-05-22
### Added
- **Project Scaffolding**: Initial Typer CLI skeletal commands.
- **Dual-mode Configuration**: Structured settings utilizing YAML and environment variables.
- **Unified LLM Clients**: Unified client interfaces for Gemini, OpenAI, Anthropic, and Ollama.
