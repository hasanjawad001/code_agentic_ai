"""Tool registration and conversion for MCP tools."""

import asyncio
from typing import Any

from fastmcp import Client
from langchain_core.tools import StructuredTool
from pydantic import create_model


class ToolRegistry:
    """Registry for managing MCP tools.

    Handles registration of Python functions as MCP tools and
    conversion to LangChain tools for agent execution.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the tool registry.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self._langchain_tools: list[StructuredTool] = []
        self._collected_images: list[dict] = []

    @property
    def langchain_tools(self) -> list[StructuredTool]:
        """Get list of LangChain tools."""
        return self._langchain_tools.copy()

    def get_collected_images(self) -> list[dict]:
        """Get all collected images."""
        return self._collected_images.copy()

    def clear_collected_images(self) -> None:
        """Clear collected images."""
        self._collected_images.clear()

    async def load_from_mcp(self, mcp_url: str) -> list[StructuredTool]:
        """Load tools from an MCP server as LangChain tools.

        Args:
            mcp_url: URL of the MCP server

        Returns:
            List of LangChain StructuredTool instances
        """
        if self.verbose:
            print(f"Loading tools from: {mcp_url}")

        async with Client(mcp_url) as client:
            mcp_tools = await client.list_tools()

        self._langchain_tools = []
        for tool in mcp_tools:
            lc_tool = self._convert_to_langchain(tool, mcp_url)
            self._langchain_tools.append(lc_tool)

        if self.verbose:
            print(f"Loaded tools: {[t.name for t in self._langchain_tools]}")

        return self._langchain_tools

    def _convert_to_langchain(self, mcp_tool: Any, mcp_url: str) -> StructuredTool:
        """Convert MCP tool to LangChain StructuredTool.

        Args:
            mcp_tool: MCP tool definition
            mcp_url: URL of the MCP server

        Returns:
            LangChain StructuredTool instance
        """
        schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}
        args_model = self._create_args_model(schema)

        async def acall_tool(**kwargs: Any) -> Any:
            # Filter out None values so MCP tool can use its defaults
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            async with Client(mcp_url) as client:
                result = await client.call_tool(mcp_tool.name, filtered_kwargs)
                if hasattr(result, "data") and isinstance(result.data, dict):
                    data = result.data.get("result", result.data)
                    # Check for image data: {has_image: True, list_image: [...], output: ...}
                    if isinstance(data, dict) and data.get("has_image"):
                        for img in data.get("list_image", []):
                            self._collected_images.append(img)
                        return data.get("output", "Image generated")
                    return data
                return result

        def call_tool(**kwargs: Any) -> Any:
            return asyncio.run(acall_tool(**kwargs))

        return StructuredTool(
            name=mcp_tool.name,
            description=mcp_tool.description or mcp_tool.name,
            func=call_tool,
            coroutine=acall_tool,
            args_schema=args_model,
        )

    def _create_args_model(self, schema: dict[str, Any]) -> Any:
        """Create Pydantic model from JSON schema.

        Args:
            schema: JSON schema for tool arguments

        Returns:
            Pydantic model class or None if no properties
        """
        if not schema or "properties" not in schema:
            return None

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        fields: dict[str, Any] = {}
        for name, prop in properties.items():
            prop_type = type_map.get(prop.get("type", "string"), str)
            if name in required:
                fields[name] = (prop_type, ...)
            elif "default" in prop:
                # Use actual default value from schema
                fields[name] = (prop_type, prop["default"])
            else:
                # Optional with no default - allow None
                fields[name] = (prop_type | None, None)

        return create_model("ToolArgs", **fields)

    def format_tool_signature(self, tool: StructuredTool) -> str:
        """Format tool with its argument signature.

        Args:
            tool: LangChain StructuredTool

        Returns:
            Formatted signature string
        """
        if tool.args_schema and hasattr(tool.args_schema, "model_json_schema"):
            schema = tool.args_schema.model_json_schema()
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            args = []
            for name, prop in props.items():
                arg_type = prop.get("type", "any")
                if name in required:
                    args.append(f"{name}: {arg_type}")
                else:
                    default = prop.get("default", "None")
                    args.append(f"{name}: {arg_type} = {default}")
            args_str = ", ".join(args)
            return f"{tool.name}({args_str}): {tool.description}"
        return f"{tool.name}(): {tool.description}"
