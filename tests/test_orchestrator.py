"""Tests for AgenticAIOrchestrator and SharedState."""

from unittest.mock import AsyncMock, patch

import pytest

from agentic_ai_mcp import AgenticAIClient, AgenticAIOrchestrator, SharedState


class TestSharedState:
    """Tests for the SharedState class."""

    def test_init_empty(self):
        state = SharedState()
        assert state.to_dict() == {}

    def test_init_with_data(self):
        state = SharedState({"key": "value"})
        assert state.get("key") == "value"

    def test_get_set(self):
        state = SharedState()
        state.set("x", 42)
        assert state.get("x") == 42

    def test_get_default(self):
        state = SharedState()
        assert state.get("missing") is None
        assert state.get("missing", "fallback") == "fallback"

    def test_contains(self):
        state = SharedState({"a": 1})
        assert "a" in state
        assert "b" not in state

    def test_to_dict_returns_copy(self):
        state = SharedState({"a": 1})
        d = state.to_dict()
        d["b"] = 2
        assert "b" not in state

    def test_repr(self):
        state = SharedState({"x": 1})
        assert repr(state) == "SharedState({'x': 1})"


class TestAgenticAIClientOrchestrationParams:
    """Tests for the new role, tool_filter, and shared_state params on AgenticAIClient."""

    def test_defaults(self):
        client = AgenticAIClient()
        assert client.role is None
        assert client._tool_filter is None
        assert client.shared_state is None

    def test_role(self):
        client = AgenticAIClient(role="researcher")
        assert client.role == "researcher"

    def test_tool_filter(self):
        client = AgenticAIClient(tool_filter=["search", "fetch"])
        assert client._tool_filter == ["search", "fetch"]

    def test_shared_state_property(self):
        client = AgenticAIClient()
        state_dict = {"key": "value"}
        client.shared_state = state_dict
        assert client.shared_state == {"key": "value"}

    def test_backward_compatible(self):
        """Existing code without role/tool_filter still works."""
        client = AgenticAIClient(
            name="test",
            mcp_url="http://localhost:8888/mcp",
            provider="anthropic",
            verbose=True,
        )
        assert client.name == "test"
        assert client.role is None
        assert client._tool_filter is None


class TestAgenticAIOrchestrator:
    """Tests for the AgenticAIOrchestrator class."""

    def test_init_requires_clients(self):
        with pytest.raises(ValueError, match="At least one client"):
            AgenticAIOrchestrator(clients=[])

    def test_init_defaults(self):
        client = AgenticAIClient(role="worker")
        orch = AgenticAIOrchestrator(clients=[client])

        assert len(orch.clients) == 1
        assert orch._flow_type == "sequential"
        assert orch._synthesizer is None
        assert isinstance(orch.shared_state, SharedState)

    def test_shared_state_attached_to_clients(self):
        c1 = AgenticAIClient(role="a")
        c2 = AgenticAIClient(role="b")
        state = SharedState({"init": True})
        AgenticAIOrchestrator(clients=[c1, c2], shared_state=state)

        # Both clients share the same underlying dict
        assert c1.shared_state is c2.shared_state
        assert c1.shared_state["init"] is True

    def test_shared_state_attached_to_synthesizer(self):
        c1 = AgenticAIClient(role="worker")
        synth = AgenticAIClient(role="synthesizer")
        state = SharedState()
        AgenticAIOrchestrator(clients=[c1], synthesizer=synth, shared_state=state)

        assert synth.shared_state is c1.shared_state

    def test_build_prompt_with_role(self):
        client = AgenticAIClient(role="researcher")
        orch = AgenticAIOrchestrator(clients=[client])

        prompt = orch._build_prompt(client, "Find info")
        assert "You are a researcher." in prompt
        assert "Find info" in prompt

    def test_build_prompt_without_role(self):
        client = AgenticAIClient()
        orch = AgenticAIOrchestrator(clients=[client])

        prompt = orch._build_prompt(client, "Do task")
        assert "You are a" not in prompt
        assert "Do task" in prompt

    def test_build_prompt_with_context(self):
        client = AgenticAIClient(role="writer")
        orch = AgenticAIOrchestrator(clients=[client])

        prompt = orch._build_prompt(client, "Write report", "Previous findings here")
        assert "You are a writer." in prompt
        assert "Previous agent results:" in prompt
        assert "Previous findings here" in prompt
        assert "Write report" in prompt

    def test_format_results(self):
        client = AgenticAIClient()
        orch = AgenticAIOrchestrator(clients=[client])

        results = [
            {"role": "researcher", "output": "Found data"},
            {"role": "writer", "output": "Wrote report"},
        ]
        formatted = orch._format_results(results)
        assert "[researcher]" in formatted
        assert "Found data" in formatted
        assert "[writer]" in formatted
        assert "Wrote report" in formatted

    async def test_run_sequential(self):
        c1 = AgenticAIClient(role="first")
        c2 = AgenticAIClient(role="second")
        orch = AgenticAIOrchestrator(clients=[c1, c2], flow_type="sequential")

        with (
            patch.object(c1, "run", new_callable=AsyncMock, return_value="result1"),
            patch.object(c2, "run", new_callable=AsyncMock, return_value="result2"),
        ):
            result = await orch.run("Do something")

        assert "[first]" in result
        assert "result1" in result
        assert "[second]" in result
        assert "result2" in result

    async def test_run_sequential_passes_context(self):
        c1 = AgenticAIClient(role="first")
        c2 = AgenticAIClient(role="second")
        orch = AgenticAIOrchestrator(clients=[c1, c2], flow_type="sequential")

        with (
            patch.object(c1, "run", new_callable=AsyncMock, return_value="output_from_first"),
            patch.object(c2, "run", new_callable=AsyncMock, return_value="final") as mock_c2,
        ):
            await orch.run("task")

        # c2 should have received c1's output as context
        called_prompt = mock_c2.call_args[0][0]
        assert "output_from_first" in called_prompt

    async def test_run_parallel(self):
        c1 = AgenticAIClient(role="alpha")
        c2 = AgenticAIClient(role="beta")
        orch = AgenticAIOrchestrator(clients=[c1, c2], flow_type="parallel")

        with (
            patch.object(c1, "run", new_callable=AsyncMock, return_value="alpha_result"),
            patch.object(c2, "run", new_callable=AsyncMock, return_value="beta_result"),
        ):
            result = await orch.run("Do both")

        assert "[alpha]" in result
        assert "alpha_result" in result
        assert "[beta]" in result
        assert "beta_result" in result

    async def test_run_with_synthesizer(self):
        c1 = AgenticAIClient(role="worker")
        synth = AgenticAIClient(role="synthesizer")
        orch = AgenticAIOrchestrator(
            clients=[c1], flow_type="sequential", synthesizer=synth
        )

        with (
            patch.object(c1, "run", new_callable=AsyncMock, return_value="raw data"),
            patch.object(
                synth, "run", new_callable=AsyncMock, return_value="synthesized"
            ) as mock_synth,
        ):
            result = await orch.run("task")

        # Synthesizer should get worker's output as context
        synth_prompt = mock_synth.call_args[0][0]
        assert "raw data" in synth_prompt
        assert result == "synthesized"

    async def test_run_with_planning(self):
        c1 = AgenticAIClient(role="planner")
        orch = AgenticAIOrchestrator(clients=[c1])

        with patch.object(
            c1, "run_with_planning", new_callable=AsyncMock, return_value="planned result"
        ):
            result = await orch.run_with_planning("complex task")

        assert "planned result" in result

    async def test_run_with_planning_and_synthesizer(self):
        c1 = AgenticAIClient(role="worker")
        synth = AgenticAIClient(role="synthesizer")
        orch = AgenticAIOrchestrator(clients=[c1], synthesizer=synth)

        with (
            patch.object(
                c1, "run_with_planning", new_callable=AsyncMock, return_value="planned"
            ),
            patch.object(
                synth,
                "run_with_planning",
                new_callable=AsyncMock,
                return_value="synth_planned",
            ),
        ):
            result = await orch.run_with_planning("complex task")

        assert result == "synth_planned"

    async def test_shared_state_updated_during_run(self):
        c1 = AgenticAIClient(role="worker")
        state = SharedState()
        orch = AgenticAIOrchestrator(clients=[c1], shared_state=state)

        with patch.object(c1, "run", new_callable=AsyncMock, return_value="done"):
            await orch.run("task")

        # agent_results should be stored in shared state
        assert "agent_results" in state
        assert state.get("agent_results")[0]["output"] == "done"

    async def test_single_client(self):
        c1 = AgenticAIClient(role="solo")
        orch = AgenticAIOrchestrator(clients=[c1])

        with patch.object(c1, "run", new_callable=AsyncMock, return_value="solo result"):
            result = await orch.run("do it")

        assert "solo result" in result

    def test_run_sync(self):
        c1 = AgenticAIClient(role="worker")
        orch = AgenticAIOrchestrator(clients=[c1])

        with patch.object(c1, "run", new_callable=AsyncMock, return_value="sync result"):
            result = orch.run_sync("task")

        assert "sync result" in result
