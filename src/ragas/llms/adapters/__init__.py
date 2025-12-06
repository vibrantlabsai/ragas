import typing as t

from ragas.llms.adapters.instructor import InstructorAdapter
from ragas.llms.adapters.litellm import LiteLLMAdapter

ADAPTERS = {
    "instructor": InstructorAdapter(),
    "litellm": LiteLLMAdapter(),
}


def get_adapter(name: str) -> t.Any:
    """
    Get adapter by name.

    Args:
        name: Adapter name ("instructor" or "litellm")

    Returns:
        StructuredOutputAdapter instance

    Raises:
        ValueError: If adapter name is unknown
    """
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name]


def auto_detect_adapter(client: t.Any, provider: str) -> str:
    """
    Auto-detect best adapter for client/provider combination.

    Logic:
    1. If client is from litellm module → use litellm
    2. If provider is gemini/google → use litellm
    3. Default → use instructor

    Args:
        client: Pre-initialized client
        provider: Provider name

    Returns:
        Adapter name ("instructor" or "litellm")
    """
    # Check if client is LiteLLM
    if hasattr(client, "__class__"):
        if "litellm" in client.__class__.__module__:
            return "litellm"

    # Check provider
    if provider.lower() in ("google", "gemini"):
        return "litellm"

    # Default
    return "instructor"


__all__ = [
    "get_adapter",
    "auto_detect_adapter",
    "ADAPTERS",
]
