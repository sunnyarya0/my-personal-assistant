# Personal AI Agent Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform Support](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](#)

An autonomous, highly extensible personal AI agent assistant designed to streamline your workflows, manage local knowledge, automate system tasks, and orchestrate complex operations.

---

## 🌟 Key Features

- **Autonomous Orchestration**: Dynamically decomposes complex objectives into achievable sub-tasks, executing them iteratively.
- **Extensible Skill Registry**: Easily register new tools and APIs (file operations, browser automation, web search, system scripts) for the agent to use.
- **Semantic Memory**: Utilizes a local vector store or lightweight database for long-term knowledge retention, context awareness, and user preferences retrieval.
- **Interactive Interfaces**: Supports terminal-based execution (CLI) and an optional modern web interface for interactive task monitoring.
- **Secure by Default**: Operates locally with configurable sandboxing, system permissions, and API credential management.

---

## 🛠️ Architecture

```
my-personal-assistant/
├── core/                  # Core agent orchestration & LLM interaction loops
│   ├── agent.py           # Reasoning engine and planning controller
│   ├── memory.py          # Semantic memory management (Vector/RAG logic)
│   └── registry.py        # Centralized registry for tools & skills
├── skills/                # Built-in and user-defined capabilities
│   ├── file_ops.py        # Safe filesystem interaction tools
│   ├── web_search.py      # Integration with search APIs
│   └── custom_skill.py    # Template for developer extensions
├── config/                # Environment and configuration files
│   └── settings.yaml      # Agent preferences and safety configurations
├── main.py                # Main CLI entrypoint
└── UI/                    # Web-based tracking dashboard (optional)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: version `3.10` or higher.
- **Access Credentials**: LLM API key (e.g., Gemini, OpenAI, or Anthropic) or a configured local LLM (e.g., via Ollama).

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sunnyarya/my-personal-assistant.git
   cd my-personal-assistant
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory and add your API keys:

```env
# LLM API Settings
LLM_PROVIDER=gemini  # or openai, anthropic, ollama
API_KEY=your_api_key_here

# Agent Preferences
AGENT_NAME="Jarvis"
SAFE_MODE=true  # Prompts for permission on destructive actions
```

### Running the Assistant

Execute the main controller via the CLI:

```bash
python main.py
```

---

## 🔌 Writing Custom Skills

Adding new abilities is straightforward. Create a Python script under `skills/` and register your functions:

```python
from core.registry import register_tool

@register_tool
def fetch_weather(location: str) -> str:
    """
    Fetches the current weather for a given location.
    
    Args:
        location (str): The city or region name.
        
    Returns:
        str: Weather report summary.
    """
    # Skill implementation details...
    return f"The weather in {location} is currently sunny and 22°C."
```

The agent automatically discovers registered skills at startup and determines when to invoke them based on their docstrings and argument signatures.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](file:///Users/sunnyarya/work/my-personal-assistant/LICENSE) for more details.
