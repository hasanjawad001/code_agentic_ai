"""Tests for AgenticAIServer."""

from agentic_ai_mcp import AgenticAIServer


class TestAgenticAIServer:
    """Tests for the AgenticAIServer class."""

    def test_init_defaults(self):
        """Test server initialization with default values."""
        server = AgenticAIServer()

        assert server.name == "agentic-ai-server"
        assert server.host == "127.0.0.1"
        assert server.port == 8888
        assert server.verbose is False
        assert server.tools == []

    def test_init_custom(self):
        """Test server initialization with custom values."""
        server = AgenticAIServer(
            name="custom-server",
            host="0.0.0.0",
            port=9999,
            verbose=True,
        )

        assert server.name == "custom-server"
        assert server.host == "0.0.0.0"
        assert server.port == 9999
        assert server.verbose is True

    def test_register_tool(self):
        """Test registering tools."""
        server = AgenticAIServer()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        server.register_tool(add)
        server.register_tool(multiply)

        assert "add" in server.tools
        assert "multiply" in server.tools
        assert len(server.tools) == 2

    def test_mcp_url_property(self):
        """Test mcp_url property returns correct format."""
        server = AgenticAIServer(host="192.168.1.100", port=8080)

        assert server.mcp_url == "http://192.168.1.100:8080/mcp"

    def test_mcp_url_property_default(self):
        """Test mcp_url property with default values."""
        server = AgenticAIServer()

        assert server.mcp_url == "http://127.0.0.1:8888/mcp"

    def test_is_running_initially_false(self):
        """Test is_running is False initially."""
        server = AgenticAIServer()

        assert server.is_running is False

    def test_start_and_stop(self):
        """Test starting and stopping the server."""
        server = AgenticAIServer(port=8895)

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        server.register_tool(add)
        server.start()

        assert server.is_running is True

        server.stop()

        assert server.is_running is False
