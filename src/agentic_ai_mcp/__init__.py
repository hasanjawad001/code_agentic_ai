"""Agentic AI MCP - Lightweight agentic AI with MCP tools."""

from agentic_ai_mcp.client import AgenticAIClient
from agentic_ai_mcp.config import Settings
from agentic_ai_mcp.orchestrator import AgenticAIOrchestrator
from agentic_ai_mcp.server import AgenticAIServer
from agentic_ai_mcp.shared_state import SharedState

__version__ = "0.6.4"
__all__ = [
    "AgenticAIServer",
    "AgenticAIClient",
    "AgenticAIOrchestrator",
    "SharedState",
    "Settings",
    "__version__",
]
