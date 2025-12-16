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


def _is_new_google_genai_client(client: t.Any) -> bool:
    """Check if client is from the new google-genai SDK.

    The new SDK (google-genai) uses genai.Client() while the old SDK
    (google-generativeai) uses genai.GenerativeModel().

    Note: The old SDK is deprecated (support ends Aug 2025). The new SDK
    is recommended but has a known upstream instructor issue with safety
    settings. See: https://github.com/567-labs/instructor/issues/1658
    """
    client_module = getattr(client, "__module__", "") or ""
    client_class = client.__class__.__name__

    # New SDK: google.genai.client.Client
    if "google.genai" in client_module and "generativeai" not in client_module:
        return True

    # Check class name as fallback (new SDK uses Client with models attribute)
    if client_class == "Client" and hasattr(client, "models"):
        return True

    return False


def auto_detect_adapter(client: t.Any, provider: str) -> str:
    """
    Auto-detect best adapter for client/provider combination.

    Logic:
    1. If client is from litellm module → use litellm
    2. If provider is gemini/google with new SDK (google-genai) → use instructor
    3. If provider is gemini/google with old SDK → use litellm
    4. Default → use instructor

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

    # Check provider for Google/Gemini
    if provider.lower() in ("google", "gemini"):
        # New google-genai SDK supports instructor natively via from_genai()
        # WARNING: Known upstream issue with instructor sending invalid safety
        # settings (HARM_CATEGORY_JAILBREAK). Track: github.com/567-labs/instructor/issues/1658
        # Workaround: Use OpenAI-compatible endpoint with Gemini base URL instead.
        if _is_new_google_genai_client(client):
            return "instructor"
        # Old SDK (deprecated, support ends Aug 2025) uses litellm
        return "litellm"

    # Default
    return "instructor"


__all__ = [
    "get_adapter",
    "auto_detect_adapter",
    "ADAPTERS",
]
