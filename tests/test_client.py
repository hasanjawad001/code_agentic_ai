"""Tests for AgenticAIClient."""

from agentic_ai_mcp import AgenticAIClient, Settings


class TestAgenticAIClient:
    """Tests for the AgenticAIClient class."""

    def test_init_defaults(self):
        """Test client initialization with default values."""
        client = AgenticAIClient()

        assert client.name == "agentic-ai-client"
        assert client.mcp_url == "http://127.0.0.1:8888/mcp"
        assert client.mcp_urls == ["http://127.0.0.1:8888/mcp"]
        assert client.model == "claude-haiku-4-5-20251001"
        assert client.verbose is False
        assert client.tools == []

    def test_init_custom(self):
        """Test client initialization with custom values."""
        client = AgenticAIClient(
            name="custom-client",
            mcp_url="http://192.168.1.100:9999/mcp",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            verbose=True,
        )

        assert client.name == "custom-client"
        assert client.mcp_url == "http://192.168.1.100:9999/mcp"
        assert client.mcp_urls == ["http://192.168.1.100:9999/mcp"]
        assert client.model == "claude-sonnet-4-20250514"
        assert client.verbose is True

    def test_init_with_mcp_urls(self):
        """Test client initialization with multiple MCP URLs."""
        urls = [
            "http://server1:8888/mcp",
            "http://server2:9999/mcp",
        ]
        client = AgenticAIClient(mcp_urls=urls)

        assert client.mcp_urls == urls
        assert client.mcp_url == "http://server1:8888/mcp"

    def test_init_with_mcp_url_and_mcp_urls(self):
        """Test that mcp_url is prepended to mcp_urls when both provided."""
        client = AgenticAIClient(
            mcp_url="http://primary:8888/mcp",
            mcp_urls=[
                "http://server1:8888/mcp",
                "http://server2:9999/mcp",
            ],
        )

        assert client.mcp_url == "http://primary:8888/mcp"
        assert client.mcp_urls == [
            "http://primary:8888/mcp",
            "http://server1:8888/mcp",
            "http://server2:9999/mcp",
        ]

    def test_init_with_mcp_url_and_mcp_urls_dedup(self):
        """Test that duplicates are removed when mcp_url is already in mcp_urls."""
        client = AgenticAIClient(
            mcp_url="http://server1:8888/mcp",
            mcp_urls=[
                "http://server1:8888/mcp",
                "http://server2:9999/mcp",
            ],
        )

        assert client.mcp_url == "http://server1:8888/mcp"
        assert client.mcp_urls == [
            "http://server1:8888/mcp",
            "http://server2:9999/mcp",
        ]

    def test_mcp_urls_returns_copy(self):
        """Test that mcp_urls returns a copy, not the internal list."""
        client = AgenticAIClient(mcp_urls=["http://server1:8888/mcp"])
        urls = client.mcp_urls
        urls.append("http://extra:8888/mcp")

        assert len(client.mcp_urls) == 1

    def test_init_with_api_key_anthropic(self):
        """Test client initialization with API key override for Anthropic."""
        client = AgenticAIClient(
            provider="anthropic",
            api_key="test-anthropic-key",
        )

        # The settings should have the overridden key
        assert client._settings.anthropic_api_key == "test-anthropic-key"

    def test_init_with_api_key_openai(self):
        """Test client initialization with API key override for OpenAI."""
        client = AgenticAIClient(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-openai-key",
        )

        # The settings should have the overridden key
        assert client._settings.openai_api_key == "test-openai-key"

    def test_init_with_settings(self):
        """Test client initialization with custom Settings."""
        settings = Settings(
            anthropic_api_key="custom-key",
            default_model="claude-sonnet-4-20250514",
            max_retries=10,
        )
        client = AgenticAIClient(settings=settings)

        assert client._settings.anthropic_api_key == "custom-key"
        assert client._settings.max_retries == 10
