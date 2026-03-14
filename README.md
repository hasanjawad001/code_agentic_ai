# Agentic AI MCP

Lightweight agentic AI with MCP tools. Supports multiple LLM providers (Anthropic, OpenAI), multi-agent orchestration, and distributed setups where tools run on one machine and agents on another.

## Install

```bash
pip install agentic-ai-mcp
```

## Setup

Set your API key in `.env` file (only needed on the client/agent machine):

```bash
# For Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
OPENAI_API_KEY=sk-...
```

## Quick Start

See the example notebooks:
- [`examples/quickstart_server.ipynb`](examples/quickstart_server.ipynb) - Run on machine exposing tools
- [`examples/quickstart_client.ipynb`](examples/quickstart_client.ipynb) - Run on machine executing agents (single & multi-agent)

## Usage

### Server (expose tools)

Run this on the machine where you want to host tools:

```python
from agentic_ai_mcp import AgenticAIServer

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def greet(name: str, times: int = 1) -> str:
    """Greet someone."""
    return ("Hello, " + name + "! ") * times

# Create server and register tools
server = AgenticAIServer(host="0.0.0.0", port=8888)
server.register_tool(add)
server.register_tool(greet)

print(f"Tools: {server.tools}")
print(f"URL: {server.mcp_url}")

# Start server in background
server.start()

# ... do other things ...

# Stop when done
server.stop()
```

### Client (run agents)

Run this on another machine to connect to the server and execute agents:

```python
from agentic_ai_mcp import AgenticAIClient

# Connect to MCP server
client = AgenticAIClient(mcp_url="http://<server-ip>:8888/mcp")

# Simple agent workflow
result = await client.run("Calculate 2+1, then greet 'Alice' that many times.")
print(result)

# Planning-based workflow for complex tasks
result = await client.run_with_planning("Calculate ((0+2) + (1+1) + 1), then greet 'Bob' that many times.")
print(result)
```

### Multiple MCP Servers

Connect to tools spread across multiple servers:

```python
from agentic_ai_mcp import AgenticAIClient

# Connect to multiple MCP servers
client = AgenticAIClient(
    mcp_urls=[
        "http://<server-1>:8888/mcp",  # math tools
        "http://<server-2>:9999/mcp",  # greeting tools
    ]
)

# The agent can use tools from all servers
result = await client.run("Calculate 2+3, then greet 'Alice' that many times.")
print(result)
```

### Multi-Agent Orchestration

Use `AgenticAIOrchestrator` to coordinate multiple agents working together on a task. Each agent can have a `role` and a `tool_filter` to specialize its behavior.

#### Sequential Flow

Agents run one after another. Each agent sees the previous agent's output as context:

```python
from agentic_ai_mcp import AgenticAIClient, AgenticAIOrchestrator

researcher = AgenticAIClient(
    mcp_url="http://<server-ip>:8888/mcp",
    role="researcher",
    tool_filter=["search"],  # only load the 'search' tool
)

writer = AgenticAIClient(
    mcp_url="http://<server-ip>:8888/mcp",
    role="writer",
    tool_filter=["write_file"],
)

orchestrator = AgenticAIOrchestrator(
    clients=[researcher, writer],
    flow_type="sequential",
)

result = await orchestrator.run("Research Python best practices and write a summary.")
print(result)
```

#### Parallel Flow

Agents run concurrently and their results are combined:

```python
orchestrator = AgenticAIOrchestrator(
    clients=[agent_a, agent_b],
    flow_type="parallel",
)

result = await orchestrator.run("Analyze this dataset from two perspectives.")
print(result)
```

#### With a Synthesizer

Add a synthesizer agent that runs last to combine all results into a final output:

```python
from agentic_ai_mcp import AgenticAIClient, AgenticAIOrchestrator

researcher = AgenticAIClient(mcp_url=MCP_URL, role="researcher")
analyst = AgenticAIClient(mcp_url=MCP_URL, role="analyst")
synthesizer = AgenticAIClient(mcp_url=MCP_URL, role="synthesizer")

orchestrator = AgenticAIOrchestrator(
    clients=[researcher, analyst],
    flow_type="parallel",
    synthesizer=synthesizer,
)

result = await orchestrator.run("Evaluate the pros and cons of microservices.")
print(result)
```

#### With Planning

The orchestrator also supports the planning workflow on each agent:

```python
result = await orchestrator.run_with_planning("Complex multi-step task...")
```

### Shared State

When orchestrating multiple agents, use `SharedState` to share data between them:

```python
from agentic_ai_mcp import AgenticAIOrchestrator, SharedState

state = SharedState({"project": "demo"})

orchestrator = AgenticAIOrchestrator(
    clients=[agent_a, agent_b],
    shared_state=state,
)

result = await orchestrator.run("Do the task.")

# After execution, agent results are stored in shared state
print(state.get("agent_results"))
print(state.to_dict())
```

### Using OpenAI

```python
from agentic_ai_mcp import AgenticAIClient

# Use OpenAI instead of Anthropic
client = AgenticAIClient(
    mcp_url="http://<server-ip>:8888/mcp",
    provider="openai",
    model="gpt-4o-mini"
)

result = await client.run("Calculate -1+2")
```

### Passing API Key Directly

```python
from agentic_ai_mcp import AgenticAIClient

# Pass API key directly (instead of using .env)
client = AgenticAIClient(
    mcp_url="http://<server-ip>:8888/mcp",
    api_key="sk-ant-..."
)
```

### Synchronous Usage

All async methods have synchronous variants for use outside of async contexts:

```python
# Single client
result = client.run_sync("Calculate 2+3")
result = client.run_with_planning_sync("Complex task...")

# Orchestrator
result = orchestrator.run_sync("Do the task.")
result = orchestrator.run_with_planning_sync("Complex task...")
```

## API Reference

### AgenticAIServer

| Property/Method | Description |
|-----------------|-------------|
| `server.tools` | List of registered tool names |
| `server.mcp_url` | Server URL |
| `server.is_running` | Check if server is running |
| `server.register_tool(func)` | Register a function as an MCP tool |
| `server.start()` | Start MCP server in background |
| `server.stop()` | Stop MCP server |

### AgenticAIClient

| Property/Method | Description |
|-----------------|-------------|
| `client.mcp_url` | Primary MCP server URL (first in the list) |
| `client.mcp_urls` | List of all MCP server URLs |
| `client.tools` | List of loaded tool names from all servers |
| `client.role` | Agent's role description (used by orchestrator) |
| `client.shared_state` | Shared state dict (set by orchestrator, `None` if standalone) |
| `client.run(prompt)` | Simple agentic workflow (async) |
| `client.run_with_planning(prompt)` | Planning-based workflow (async) |
| `client.run_sync(prompt)` | Simple agentic workflow (sync) |
| `client.run_with_planning_sync(prompt)` | Planning-based workflow (sync) |

### AgenticAIOrchestrator

| Property/Method | Description |
|-----------------|-------------|
| `orchestrator.clients` | List of orchestrated clients |
| `orchestrator.shared_state` | The SharedState instance |
| `orchestrator.run(prompt)` | Run with simple ReAct on each client (async) |
| `orchestrator.run_with_planning(prompt)` | Run with planning on each client (async) |
| `orchestrator.run_sync(prompt)` | Synchronous version of `run()` |
| `orchestrator.run_with_planning_sync(prompt)` | Synchronous version of `run_with_planning()` |

### SharedState

| Property/Method | Description |
|-----------------|-------------|
| `state.get(key, default)` | Get a value from shared state |
| `state.set(key, value)` | Set a value in shared state |
| `state.to_dict()` | Return a copy of the full state |
| `key in state` | Check if a key exists |

## License

MIT
