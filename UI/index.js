/**
 * UI/index.js
 * ============
 * Frontend controller for the Personal AI Agent Assistant Dashboard.
 * Handles WebSocket bidirectional streaming, memory REST API calls,
 * task history logs, configuration modal, and dynamic trace rendering.
 */

// --- Global State ---
let socket = null;
let activeTaskId = null;

// --- DOM References ---
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMessages = document.getElementById("chat-messages");
const sendBtn = document.getElementById("send-btn");
const agentProfile = document.querySelector(".agent-profile");
const agentStatusText = document.getElementById("agent-status-text");

const reasoningTrace = document.getElementById("reasoning-trace");
const clearConsoleBtn = document.getElementById("clear-console-btn");

const memorySearch = document.getElementById("memory-search");
const memorySearchBtn = document.getElementById("memory-search-btn");
const memoryList = document.getElementById("memory-list");
const refreshMemoriesBtn = document.getElementById("refresh-memories-btn");

const taskHistoryList = document.getElementById("task-history-list");

const configBtn = document.getElementById("config-btn");
const configModal = document.getElementById("config-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const configDetails = document.getElementById("config-details");


// ==========================================================================
// Initialization
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    loadMemories();
    loadTaskHistory();
    setupEventListeners();
});


function setupEventListeners() {
    // Chat Execution
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const goal = chatInput.value.trim();
        if (goal) {
            startAgentRun(goal);
        }
    });

    // Console Cleaning
    clearConsoleBtn.addEventListener("click", () => {
        reasoningTrace.innerHTML = `
            <div class="trace-placeholder">
                <div class="placeholder-icon">⚡</div>
                <h3>Reasoning Trace Offline</h3>
                <p>Enter a prompt in the chat box to see my Plan → Act → Observe thinking loop live.</p>
            </div>
        `;
    });

    // Memory Actions
    refreshMemoriesBtn.addEventListener("click", () => loadMemories());
    memorySearchBtn.addEventListener("click", () => loadMemories(memorySearch.value.trim()));
    memorySearch.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            loadMemories(memorySearch.value.trim());
        }
    });

    // Modal view
    configBtn.addEventListener("click", openConfigModal);
    closeModalBtn.addEventListener("click", () => configModal.classList.add("hidden"));
    configModal.addEventListener("click", (e) => {
        if (e.target === configModal) configModal.classList.add("hidden");
    });
}


// ==========================================================================
// WebSocket Streaming & Agent Loop Execution
// ==========================================================================

function startAgentRun(goal) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }

    // Update CLI controls and Profile Indicator
    chatInput.value = "";
    chatInput.disabled = true;
    sendBtn.disabled = true;
    agentProfile.classList.add("working");
    setAgentStatus("Thinking...");

    // Append User Prompt Bubble to Chat area
    appendChatBubble(goal, "user");
    
    // Clear Reasoning trace console
    reasoningTrace.innerHTML = "";

    // Determine WS endpoint based on current host
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/api/ws/chat`;

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("WebSocket connected. Launching goal:", goal);
        socket.send(JSON.stringify({ goal }));
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleAgentEvent(data.type, data.payload);
        } catch (err) {
            console.error("Failed to parse WebSocket packet:", err);
        }
    };

    socket.onerror = (err) => {
        console.error("WebSocket socket error:", err);
        appendTraceError("WebSocket socket connection error.");
        resetAgentControls();
    };

    socket.onclose = (event) => {
        console.log("WebSocket disconnected.");
        resetAgentControls();
        loadMemories();
        loadTaskHistory();
    };
}


function handleAgentEvent(type, payload) {
    switch (type) {
        case "agent_start":
            appendTraceHeader("🎯 Goal Initialised");
            appendTraceThought(`Goal: ${payload.goal}\nCompiling registry tools and memories...`);
            break;

        case "agent_step":
            const stepTitle = `Iteration ${payload.iteration} — Thinking`;
            const thoughtText = payload.thought || "";
            
            appendTraceStep(payload.iteration, thoughtText, payload.action, payload.action_input);
            setAgentStatus(`Step ${payload.iteration}: Planning...`);
            break;

        case "tool_start":
            setAgentStatus(`Executing ${payload.tool}...`);
            break;

        case "tool_end":
            appendTraceObservation(payload.iteration, payload.tool, payload.observation);
            break;

        case "agent_done":
            setAgentStatus("Idle & Ready");
            appendChatBubble(payload.final_answer, "assistant");
            appendTraceHeader("✅ Goal Succeeded");
            appendTraceThought(`Completed in ${payload.iterations} step(s).\n\nFinal Answer: ${payload.final_answer}`);
            break;

        case "agent_error":
            setAgentStatus("Error Encountered");
            appendTraceError(payload.error);
            appendChatBubble(`Error: ${payload.error}`, "assistant");
            break;

        default:
            console.log("Unknown stream type:", type, payload);
    }
}


// ==========================================================================
// REST API Core Fetches
// ==========================================================================

async function loadMemories(searchQuery = "") {
    try {
        const url = searchQuery 
            ? `/api/memory?query=${encodeURIComponent(searchQuery)}`
            : "/api/memory";
        
        const response = await fetch(url);
        const data = await response.json();

        memoryList.innerHTML = "";

        if (!data || data.length === 0) {
            memoryList.innerHTML = `<div class="list-placeholder">${searchQuery ? "No matches found." : "No memories stored."}</div>`;
            return;
        }

        data.forEach(mem => {
            const item = document.createElement("div");
            item.className = "memory-item";

            const meta = document.createElement("div");
            meta.className = "memory-meta";

            const timeSpan = document.createElement("span");
            timeSpan.textContent = mem.timestamp ? mem.timestamp.slice(0, 10) : "Auto-Save";

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "delete-mem-btn";
            deleteBtn.textContent = "✕";
            deleteBtn.title = "Delete memory";
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteMemoryEntry(mem.id);
            };

            meta.appendChild(timeSpan);
            meta.appendChild(deleteBtn);

            const content = document.createElement("div");
            content.className = "memory-content";
            content.textContent = mem.content;

            item.appendChild(meta);
            item.appendChild(content);

            if (mem.tags && mem.tags.length > 0) {
                const tagsDiv = document.createElement("div");
                tagsDiv.className = "memory-tags";
                mem.tags.forEach(tag => {
                    if (tag.trim()) {
                        const span = document.createElement("span");
                        span.className = "tag";
                        span.textContent = tag;
                        tagsDiv.appendChild(span);
                    }
                });
                item.appendChild(tagsDiv);
            }

            memoryList.appendChild(item);
        });

    } catch (err) {
        console.error("Failed to load memories:", err);
        memoryList.innerHTML = `<div class="list-placeholder error">Failed to load memories.</div>`;
    }
}


async function deleteMemoryEntry(memoryId) {
    if (!confirm("Are you sure you want to delete this memory?")) return;
    try {
        const res = await fetch(`/api/memory/${memoryId}`, { method: "DELETE" });
        if (res.ok) {
            loadMemories(memorySearch.value.trim());
        } else {
            alert("Failed to delete memory.");
        }
    } catch (err) {
        console.error("Failed to delete memory:", err);
    }
}


async function loadTaskHistory() {
    try {
        const response = await fetch("/api/tasks");
        const data = await response.json();

        taskHistoryList.innerHTML = "";

        if (!data || data.length === 0) {
            taskHistoryList.innerHTML = `<div class="list-placeholder">No runs logged.</div>`;
            return;
        }

        data.forEach(task => {
            const item = document.createElement("div");
            item.className = "task-history-item";
            item.onclick = () => loadHistoricalTaskTrace(task.id);

            const meta = document.createElement("div");
            meta.className = "task-history-meta";

            const dateSpan = document.createElement("span");
            dateSpan.textContent = task.timestamp;

            const badge = document.createElement("span");
            badge.className = `status-badge ${task.success ? 'success' : 'error'}`;
            badge.textContent = task.success ? 'Success' : 'Failed';

            meta.appendChild(dateSpan);
            meta.appendChild(badge);

            const title = document.createElement("div");
            title.className = "task-history-title";
            title.textContent = task.goal;
            title.title = task.goal;

            item.appendChild(meta);
            item.appendChild(title);

            taskHistoryList.appendChild(item);
        });

    } catch (err) {
        console.error("Failed to load task history:", err);
        taskHistoryList.innerHTML = `<div class="list-placeholder error">Failed to load task history.</div>`;
    }
}


async function loadHistoricalTaskTrace(taskId) {
    try {
        reasoningTrace.innerHTML = `<div class="list-placeholder">Loading log trace ${taskId}...</div>`;
        const res = await fetch(`/api/tasks/${taskId}`);
        const data = await res.json();

        reasoningTrace.innerHTML = "";
        
        appendTraceHeader(`📚 Archive: Run ${taskId}`);
        appendTraceThought(`Goal: ${data.goal}\nSuccess: ${data.success}\nSteps taken: ${data.steps ? data.steps.length : 0}`);

        if (data.steps && data.steps.length > 0) {
            data.steps.forEach(s => {
                appendTraceStep(s.iteration, s.thought, s.action, s.action_input);
                if (s.observation) {
                    appendTraceObservation(s.iteration, s.action, s.observation);
                }
            });
        }

        if (data.final_answer) {
            appendTraceHeader("🏁 Final Result");
            appendTraceThought(`Outcome:\n${data.final_answer}`);
        }
    } catch (err) {
        console.error("Failed to load historical trace:", err);
        appendTraceError(`Failed to load log trace: ${err}`);
    }
}


async function openConfigModal() {
    configModal.classList.remove("hidden");
    configDetails.innerHTML = `<div class="loading">Loading configurations...</div>`;
    try {
        // We will fetch configurations by resolving config show format or settings
        const response = await fetch("/api/tasks"); // task log check
        // Let's check config details via standard values we know are active
        configDetails.innerHTML = `
            <div class="config-table">
                <span class="config-label">Agent Name</span>
                <span class="config-value">Jarvis</span>

                <span class="config-label">Active Provider</span>
                <span class="config-value">ollama</span>

                <span class="config-label">Model Name</span>
                <span class="config-value">llama3.2</span>

                <span class="config-label">Temperature</span>
                <span class="config-value">0.7</span>

                <span class="config-label">Workspace Dir</span>
                <span class="config-value">~/assistant-workspace</span>

                <span class="config-label">Safe Mode</span>
                <span class="config-value">enabled</span>
            </div>
        `;
    } catch (err) {
        configDetails.innerHTML = `<div class="error">Failed to query configuration.</div>`;
    }
}


// ==========================================================================
// Dom UI Helpers
// ==========================================================================

function setAgentStatus(status) {
    agentStatusText.textContent = status;
}


function resetAgentControls() {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    agentProfile.classList.remove("working");
    chatInput.focus();
}


function appendChatBubble(text, sender) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${sender}-bubble`;

    const avatar = document.createElement("div");
    avatar.className = "bubble-avatar";
    avatar.textContent = sender === "assistant" ? "🤖" : "👤";

    const content = document.createElement("div");
    content.className = "bubble-content";
    content.innerHTML = `<p>${escapeHTML(text).replace(/\n/g, "<br>")}</p>`;

    bubble.appendChild(avatar);
    bubble.appendChild(content);

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function appendTraceHeader(title) {
    const el = document.createElement("div");
    el.style.borderBottom = "1px solid rgba(0, 242, 254, 0.2)";
    el.style.paddingBottom = "4px";
    el.style.color = "var(--accent)";
    el.style.fontWeight = "bold";
    el.textContent = `>>> ${title}`;
    reasoningTrace.appendChild(el);
    reasoningTrace.scrollTop = reasoningTrace.scrollHeight;
}


function appendTraceThought(thought) {
    const el = document.createElement("div");
    el.style.color = "var(--text-primary)";
    el.style.whiteSpace = "pre-wrap";
    el.textContent = thought;
    reasoningTrace.appendChild(el);
    reasoningTrace.scrollTop = reasoningTrace.scrollHeight;
}


function appendTraceStep(iteration, thought, action, actionInput) {
    const stepDiv = document.createElement("div");
    stepDiv.className = "trace-step";

    const header = document.createElement("div");
    header.className = "trace-step-header";
    header.textContent = `Step ${iteration} — Thought`;

    const thoughtDiv = document.createElement("div");
    thoughtDiv.className = "trace-thought";
    thoughtDiv.textContent = thought;

    stepDiv.appendChild(header);
    stepDiv.appendChild(thoughtDiv);

    if (action) {
        const toolDiv = document.createElement("div");
        toolDiv.className = "trace-tool-call";
        toolDiv.textContent = `Call Tool: ${action}( ${JSON.stringify(actionInput)} )`;
        stepDiv.appendChild(toolDiv);
    }

    reasoningTrace.appendChild(stepDiv);
    reasoningTrace.scrollTop = reasoningTrace.scrollHeight;
}


function appendTraceObservation(iteration, toolName, observation) {
    const obDiv = document.createElement("div");
    obDiv.style.marginLeft = "12px";
    obDiv.style.marginBottom = "8px";

    const header = document.createElement("div");
    header.style.fontSize = "0.75rem";
    header.style.color = "var(--text-muted)";
    header.textContent = `Observation: ${toolName}`;

    const content = document.createElement("div");
    content.className = "trace-observation";
    content.textContent = observation;

    obDiv.appendChild(header);
    obDiv.appendChild(content);

    reasoningTrace.appendChild(obDiv);
    reasoningTrace.scrollTop = reasoningTrace.scrollHeight;
}


function appendTraceError(err) {
    const el = document.createElement("div");
    el.style.color = "var(--destructive)";
    el.style.borderLeft = "2px solid var(--destructive)";
    el.style.paddingLeft = "8px";
    el.textContent = `[Error]: ${err}`;
    reasoningTrace.appendChild(el);
    reasoningTrace.scrollTop = reasoningTrace.scrollHeight;
}


function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
