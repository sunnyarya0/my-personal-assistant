# Personal AI Agent Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform Support](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](#)
[![Version](https://img.shields.io/badge/release-v1.0.0-emerald)](#)

An autonomous, 100% local, highly extensible personal AI agent assistant designed to streamline your workflows, manage local knowledge, automate system tasks, and orchestrate complex operations. Powered by local LLMs (Ollama) and local vector stores (ChromaDB), it guarantees complete privacy and secure, robust automation.

---

## 🌟 Key Features

- **Autonomous Reasoning Engine (ReAct)**: Dynamically plans and executes multi-step objectives using interactive Plan → Thought → Act → Observe loops.
- **Galactic Web Dashboard**: A stunning, real-time three-column glassmorphic monitor presenting live reasoning logs, interactive chat panels, task history lookup, and a searchable semantic memory browser.
- **Offline Semantic Memory (RAG)**: Integrates ChromaDB and `sentence-transformers` for fully local embeddings and contextual memory retrieval.
- **Path-Sandboxed Workspace**: Enforces absolute filesystem security boundaries to block path traversal escapes outside the user's workspace.
- **Command Sanitization**: Built-in shell script blocklist protection guarding standard utilities against unauthorized or destructive actions (such as root access elevations).
- **Proactive Warnings & Backoff Retries**: Includes LLM token budget threshold alerts and automated exponential backoff retry logic.

---

## 🏗️ System Architecture

Refer to our complete [Architecture Blueprint](docs/ARCHITECTURE.md) for detailed descriptions of the internal logic. Below is a high-level overview:

```
my-personal-assistant/
├── core/                  # Core agent orchestration & LLM interaction loops
│   ├── agent.py           # ReAct engine, planning, and outcome redact-logging
│   ├── config.py          # Tiered configuration schema loader
│   ├── embedder.py        # Sentence-transformers local embeddings singleton
│   ├── llm_client.py      # Unified LLM provider clients with backoff retry
│   ├── memory.py          # Short & long term (ChromaDB) semantic database
│   ├── prompt.py          # ReAct system prompt and JSON parsing utilities
│   └── registry.py        # Centralized tool & skill decorator registry
├── skills/                # Discovered agent capability toolboxes
│   ├── file_ops.py        # Sandboxed file/directory operations
│   ├── web_search.py      # DuckDuckGo Lite searcher and clean HTML scraper
│   ├── system_ops.py      # Sanitized terminal run command and diagnostics
│   └── datetime_ops.py    # Local clock queries and NL date parsers
├── config/                # Settings file repository
│   └── settings.yaml      # Multi-tiered configuration profiles
├── UI/                    # FastAPI web server assets and premium CSS/JS dashboard
├── docs/                  # Documentation manuals
│   ├── ARCHITECTURE.md    # Blueprint system details
│   ├── SKILLS.md          # Guide to writing custom skills
│   └── CONFIGURATION.md   # Setup variables and schema documentation
├── CHANGELOG.md           # Traces progression to production v1.0.0
├── main.py                # Main Typer CLI entrypoint
└── pyproject.toml         # Packaging metadata, tools, and dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: version `3.10` or higher.
- **Ollama**: (Optional for 100% local operation) [Download and install Ollama](https://ollama.com/), then pull a model:
  ```bash
  ollama pull llama3.2
  ```

### 2. Installation
Setup your environment with a single command:
```bash
# Clone the repository
git clone https://github.com/sunnyarya/my-personal-assistant.git
cd my-personal-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (including developer tools)
pip install --upgrade pip
pip install -e .[dev]
```

### 3. Configuration
By default, the assistant runs on Ollama using `llama3.2`. To customize providers or safety preferences, configure a local `.env` file. See the full [Configuration Guide](docs/CONFIGURATION.md) for details.

```env
# Choose provider and set API keys
LLM_PROVIDER=ollama         # ollama | gemini | openai | anthropic
LLM_MODEL=llama3.2          # e.g., llama3.2 or gemini-2.0-flash
API_KEY=                    # Required only for cloud APIs

# Workspace configuration
WORKSPACE_DIR=~/assistant-workspace
SAFE_MODE=true

# Scraper Domain Scoping
ALLOWED_DOMAINS=wikipedia.org,github.com
```

### 4. Running the Dashboard (Web UI)
Launch the server and watch your agent think in real time:
```bash
python main.py serve
```
Uvicorn will spin up a local server on `http://127.0.0.1:8000` and automatically open your default web browser.

### 5. Running via CLI
You can also run natural-language goals directly inside your shell:
```bash
# Simple execution
python main.py chat "Create a text file names.txt inside the workspace."

# Verbose reasoning trace
python main.py chat -v "Search the web for Python optimization tips."
```

---

## 🔌 Custom Skills Ecosystem

Extending your agent is extremely simple. Write a Python function under `skills/` and register it using the decorator:

```python
from core.registry import register_tool

@register_tool(description="Calculate body mass index.")
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    Calculate and return BMI index report.

    Args:
        weight_kg: Weight of the person in kilograms.
        height_m: Height of the person in meters.
    """
    bmi = weight_kg / (height_m ** 2)
    return f"Calculated BMI is {bmi:.1f}."
```
Learn best-practices, path sandboxing, and destructive safe-mode tagging in the [Developer Skills Manual](docs/SKILLS.md).

---

## 🧪 Verification & Development

Run our full test suite containing unit, mock-integration, and REST/WebSocket API endpoints checks:
```bash
# Run pytest with coverage mapping
pytest
```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
