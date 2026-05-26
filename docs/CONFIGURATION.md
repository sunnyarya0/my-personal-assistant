# Configuration Guide

The Personal AI Agent Assistant supports a tiered configuration hierarchy. Settings are resolved in the following priority order (highest to lowest):
1. **Environment Variables** (e.g. `LLM_PROVIDER`, `API_KEY`)
2. **`.env` File**
3. **`config/settings.yaml`**
4. **Built-in Defaults**

---

## 1. Schema Specifications

Here is a complete reference of all available configuration parameters, their types, and descriptions:

| Parameter | YAML Mapping | Environment Variable | Default Value | Description |
|-----------|--------------|----------------------|---------------|-------------|
| **LLM Provider** | `llm.provider` | `LLM_PROVIDER` | `"gemini"` | The LLM engine to use (`gemini`, `openai`, `anthropic`, `ollama`). |
| **LLM Model** | `llm.model` | `LLM_MODEL` | *Provider specific* | The model name (e.g. `gemini-2.0-flash`, `llama3.2`). |
| **API Key** | *N/A* | `API_KEY` | `""` | The API key credentials (not required for Ollama). |
| **Temperature** | `llm.temperature` | `LLM_TEMPERATURE` | `0.7` | Sampling temperature for LLM chat generation (0.0 to 2.0). |
| **Max Tokens** | `llm.max_tokens` | `LLM_MAX_TOKENS` | `8192` | The maximum generation output limits. |
| **Ollama URL** | `llm.ollama_base_url`| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Local URL host of your Ollama process. |
| **Agent Name** | `agent.name` | `AGENT_NAME` | `"Jarvis"` | The name the agent uses to identify itself in chats. |
| **Safe Mode** | `agent.safe_mode` | `SAFE_MODE` | `true` | Intercepts destructive tools to ask user y/n confirmation. |
| **Max Iterations**| `agent.max_iterations`| `MAX_ITERATIONS` | `10` | Loop ceiling to halt reasoning in case of infinite runs. |
| **Verbose** | `agent.verbose` | `AGENT_VERBOSE` | `false` | Outputs ReAct thought steps live to the terminal console. |
| **Workspace Dir** | `storage.workspace_dir`| `WORKSPACE_DIR` | `~/assistant-workspace`| Directory sandbox where files operations are permitted. |
| **Log Directory** | `storage.log_dir` | `LOG_DIR` | `~/.assistant/logs` | JSON task logs storage folder. |
| **UI Port** | `ui.port` | `UI_PORT` | `8000` | Port used by the FastAPI backend server. |
| **Auto Browser** | `ui.auto_open_browser`| *N/A* | `true` | Starts serve command and immediately launches system browser. |
| **Allowed Domains**| `security.allowed_domains`| `ALLOWED_DOMAINS`| `[]` | Domain restrictions list for web scraping and fetching. |
| **Rate Limit** | `security.rate_limit_rpm`| `RATE_LIMIT_RPM`| `60` | Client requests per minute allowed on `/api` routes. |

---

## 2. YAML Structure Example

Here is a standard `config/settings.yaml` configuration profile:

```yaml
agent:
  name: "Jarvis"
  safe_mode: true
  max_iterations: 10
  verbose: false

llm:
  provider: "ollama"
  model: "llama3.2"
  temperature: 0.7
  max_tokens: 8192
  ollama_base_url: "http://localhost:11434"

storage:
  workspace_dir: "~/assistant-workspace"
  log_dir: "~/.assistant/logs"
  memory_dir: "~/.assistant/memory"

ui:
  port: 8000
  auto_open_browser: true

security:
  allowed_domains:
    - "wikipedia.org"
    - "github.com"
  rate_limit_rpm: 60
```

---

## 3. Environment Variable Configuration (`.env`)

For production deployments or local overrides, create a `.env` file in the root directory:

```env
# Choose provider and set matching API keys
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
API_KEY=AIzaSyD-YourGeminiKeyHere

# Tighten loops and folders
SAFE_MODE=true
MAX_ITERATIONS=6
WORKSPACE_DIR=/Users/user/my-workdir

# Security policies
ALLOWED_DOMAINS=wikipedia.org,github.com
RATE_LIMIT_RPM=30
```
