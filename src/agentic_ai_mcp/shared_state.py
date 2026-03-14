"""SharedState - A simple shared state container for orchestrated agents."""

from typing import Any


class SharedState:
    """Shared state that multiple AgenticAIClient instances can read and write.

    Used by AgenticAIOrchestrator to give all agents access to a common
    key-value store during execution.

    Example:
        state = SharedState({"context": "initial data"})
        state.set("result", 42)
        state.get("result")  # 42
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        """Initialize SharedState.

        Args:
            initial: Optional initial data to populate the state.
        """
        self._data: dict[str, Any] = dict(initial) if initial else {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared state.

        Args:
            key: The key to look up.
            default: Value to return if key is not found.

        Returns:
            The value, or default if not found.
        """
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the shared state.

        Args:
            key: The key to set.
            value: The value to store.
        """
        self._data[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the full state as a dict."""
        return self._data.copy()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"SharedState({self._data!r})"
