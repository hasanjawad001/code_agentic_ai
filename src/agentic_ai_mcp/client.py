"""AgenticAIClient - Client for running agentic workflows with MCP tools."""

import asyncio
import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.config import Settings, get_default_model, get_settings
from agentic_ai_mcp.providers import get_provider
from agentic_ai_mcp.tools import ToolRegistry
from agentic_ai_mcp.workflows import PlanningWorkflow


class AgenticAIClient:
    """Client for running agentic workflows with MCP tools.

    Example:
        from agentic_ai_mcp import AgenticAIClient

        client = AgenticAIClient(
            mcp_url="http://127.0.0.1:8888/mcp",
            # api_key="sk-..."  # Optional if .env is set
        )

        # Simple task
        result = await client.run("Calculate 2+3")

        # Complex task with planning
        result = await client.run_with_planning("Calculate X, then do Y")
    """

    def __init__(
        self,
        name: str = "agentic-ai-client",
        mcp_url: str | None = None,
        mcp_urls: list[str] | None = None,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        verbose: bool = False,
        settings: Settings | None = None,
        role: str | None = None,
        tool_filter: list[str] | None = None,
    ) -> None:
        """Initialize AgenticAIClient.

        Args:
            name: Name for the client
            mcp_url: URL of a single MCP server (backward compatible)
            mcp_urls: List of MCP server URLs to connect to
            provider: LLM provider ('anthropic' or 'openai')
            model: LLM model name (defaults to settings)
            api_key: API key for the provider (optional, overrides .env)
            verbose: Enable verbose output
            settings: Optional Settings override
            role: Agent's role description (e.g. "researcher", "writer").
                Used by the orchestrator to build context-aware prompts.
            tool_filter: List of tool names to load. If set, only tools
                matching these names are kept after loading from MCP.

        Note:
            You can use either ``mcp_url`` or ``mcp_urls``.
            If both are provided, ``mcp_url`` is prepended to ``mcp_urls`` (duplicates removed).
            If neither is provided, defaults to ``["http://127.0.0.1:8888/mcp"]``.
        """
        self.name = name
        self.role = role
        self._tool_filter = tool_filter
        self._shared_state: dict[str, Any] | None = None

        # Build mcp_urls list with backward compatibility
        if mcp_url is not None and mcp_urls is not None:
            # Both provided: prepend mcp_url, deduplicate preserving order
            combined = [mcp_url] + [u for u in mcp_urls if u != mcp_url]
            self._mcp_urls = combined
        elif mcp_urls is not None:
            self._mcp_urls = list(mcp_urls)
        elif mcp_url is not None:
            self._mcp_urls = [mcp_url]
        else:
            self._mcp_urls = ["http://127.0.0.1:8888/mcp"]

        self.model = model or get_default_model()
        self.verbose = verbose

        # Settings - handle api_key override
        if settings is not None:
            self._settings = settings
        else:
            self._settings = get_settings()

        # Override API key if provided
        if api_key is not None:
            if provider == "anthropic":
                # Create new settings with overridden API key
                self._settings = Settings(
                    anthropic_api_key=api_key,
                    openai_api_key=self._settings.openai_api_key,
                    default_model=self._settings.default_model,
                    default_provider=self._settings.default_provider,
                    mcp_host=self._settings.mcp_host,
                    mcp_port=self._settings.mcp_port,
                    max_retries=self._settings.max_retries,
                    retry_base_delay=self._settings.retry_base_delay,
                    retry_max_delay=self._settings.retry_max_delay,
                )
            elif provider == "openai":
                self._settings = Settings(
                    anthropic_api_key=self._settings.anthropic_api_key,
                    openai_api_key=api_key,
                    default_model=self._settings.default_model,
                    default_provider=self._settings.default_provider,
                    mcp_host=self._settings.mcp_host,
                    mcp_port=self._settings.mcp_port,
                    max_retries=self._settings.max_retries,
                    retry_base_delay=self._settings.retry_base_delay,
                    retry_max_delay=self._settings.retry_max_delay,
                )

        # Components
        self._tool_registry = ToolRegistry(verbose=verbose)

        # Provider
        self._provider = get_provider(
            provider_type=provider,
            model=self.model,
            settings=self._settings,
        )

        # Agent state
        self._agent: Any = None
        self._planning_workflow: PlanningWorkflow | None = None

    @property
    def mcp_url(self) -> str:
        """Primary MCP server URL (first in the list)."""
        return self._mcp_urls[0]

    @property
    def mcp_urls(self) -> list[str]:
        """List of all MCP server URLs."""
        return self._mcp_urls.copy()

    @property
    def tools(self) -> list[str]:
        """List of loaded tool names from all MCP servers."""
        return [t.name for t in self._tool_registry.langchain_tools]

    @property
    def shared_state(self) -> dict[str, Any] | None:
        """Shared state dict set by the orchestrator. None if not orchestrated."""
        return self._shared_state

    @shared_state.setter
    def shared_state(self, state: dict[str, Any] | None) -> None:
        """Set the shared state (called by the orchestrator)."""
        self._shared_state = state

    def _get_llm(self) -> Any:
        """Get the LLM instance from the provider."""
        return self._provider.get_chat_model()

    def _display_collected_images(self) -> None:
        """Display any images collected during tool execution."""
        images = self._tool_registry.get_collected_images()
        if not images:
            return

        try:
            # Try Jupyter display first
            from IPython.display import Image as IPImage
            from IPython.display import display

            for img_data in images:
                img_bytes = base64.b64decode(img_data["data"])
                display(IPImage(data=img_bytes))
                if self.verbose:
                    w, h = img_data.get("width", "?"), img_data.get("height", "?")
                    print(f"[Displayed {w}x{h} image]")
        except ImportError:
            # Fallback: save to file
            for i, img_data in enumerate(images):
                filename = f"output_image_{i}.{img_data.get('format', 'png')}"
                img_bytes = base64.b64decode(img_data["data"])
                Path(filename).write_bytes(img_bytes)
                if self.verbose:
                    print(f"[Saved image to {filename}]")

        self._tool_registry.clear_collected_images()

    async def _load_tools(self) -> None:
        """Load tools from all MCP servers as LangChain tools.

        If ``tool_filter`` was set, only tools whose names appear in the
        filter list are kept.
        """
        await self._tool_registry.load_from_mcp_urls(self._mcp_urls)

        if self._tool_filter is not None:
            allowed = set(self._tool_filter)
            self._tool_registry._langchain_tools = [
                t for t in self._tool_registry._langchain_tools if t.name in allowed
            ]

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt (simple ReAct workflow).

        Args:
            prompt: Task for the agent

        Returns:
            Agent's response
        """
        # Load tools if not loaded
        if not self._tool_registry.langchain_tools:
            await self._load_tools()

        # Create agent if not created
        if self._agent is None:
            self._agent = create_react_agent(
                self._get_llm(),
                self._tool_registry.langchain_tools,
            )

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"PROMPT: {prompt}")
            print(f"{'=' * 50}\n")

        # Run
        result = await self._agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        # Process
        messages = result.get("messages", [])
        final_response = "No response"
        step = 0

        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step += 1
                        if self.verbose:
                            print(f"STEP {step}: {tc['name']}({tc['args']})")
                if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    final_response = str(msg.content)
            elif isinstance(msg, ToolMessage) and self.verbose:
                print(f"  → {msg.content}\n")

        if self.verbose:
            print(f"{'=' * 50}")
            print(f"RESULT: {final_response}")
            print(f"{'=' * 50}\n")

        # Display any collected images
        self._display_collected_images()

        return final_response

    async def run_with_planning(self, prompt: str) -> str:
        """Run the agent with planning for complex tasks.

        Uses LangGraph StateGraph to:
        1. Plan: Break down the task into steps
        2. Execute: Run each step with tools
        3. Synthesize: Combine results into final response

        Args:
            prompt: Complex task for the agent

        Returns:
            Agent's response
        """
        # Load tools if not loaded
        if not self._tool_registry.langchain_tools:
            await self._load_tools()

        # Create planning workflow if not created
        if self._planning_workflow is None:
            self._planning_workflow = PlanningWorkflow(
                llm=self._get_llm(),
                tools=self._tool_registry.langchain_tools,
                tool_registry=self._tool_registry,
                max_retries=self._settings.max_retries,
                verbose=self.verbose,
            )

        result = await self._planning_workflow.run(prompt)

        # Display any collected images
        self._display_collected_images()

        return result

    def run_sync(self, prompt: str) -> str:
        """Synchronous version of run().

        Args:
            prompt: Task for the agent

        Returns:
            Agent's response
        """
        return asyncio.run(self.run(prompt))

    def run_with_planning_sync(self, prompt: str) -> str:
        """Synchronous version of run_with_planning().

        Args:
            prompt: Complex task for the agent

        Returns:
            Agent's response
        """
        return asyncio.run(self.run_with_planning(prompt))
