"""Agentic AI MCP - Lightweight agentic AI with MCP tools."""

from agentic_ai_mcp.client import AgenticAIClient
from agentic_ai_mcp.config import Settings
from agentic_ai_mcp.server import AgenticAIServer

__version__ = "0.6.3"
__all__ = ["AgenticAIServer", "AgenticAIClient", "Settings", "__version__"]
