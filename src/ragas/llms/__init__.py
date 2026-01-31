from ragas.llms.base import (
    BaseRagasLLM,
    InstructorBaseRagasLLM,
    InstructorLLM,
    InstructorTypeVar,
    LangchainLLMWrapper as _LangchainLLMWrapper,
    LlamaIndexLLMWrapper as _LlamaIndexLLMWrapper,
    llm_factory,
)
from ragas.llms.groq_wrapper import GroqLLMWrapper
from ragas.llms.haystack_wrapper import HaystackLLMWrapper
from ragas.llms.litellm_llm import LiteLLMStructuredLLM
from ragas.llms.oci_genai_wrapper import OCIGenAIWrapper, oci_genai_factory
from ragas.utils import DeprecationHelper

# Groq Integration Notes:
# - llm_factory() with provider="groq" → Uses instructor adapter for structured outputs
#   Example: llm = llm_factory("llama3-70b-8192", provider="groq", client=groq_client)
#
# - GroqLLMWrapper() → Direct wrapper for text generation (BaseRagasLLM interface)
#   Example: llm = GroqLLMWrapper(groq_client, model="llama3-70b-8192")
#
# Both approaches are valid. Use llm_factory for structured outputs (Pydantic models),
# use GroqLLMWrapper for direct text generation with custom rate limiting.

# Create deprecation wrappers for legacy classes
LangchainLLMWrapper = DeprecationHelper(
    _LangchainLLMWrapper,
    "LangchainLLMWrapper is deprecated and will be removed in a future version. "
    "Use llm_factory instead: "
    "from openai import OpenAI; "
    "from ragas.llms import llm_factory; "
    "llm = llm_factory('gpt-4o-mini', client=OpenAI(api_key='...'))",
)

LlamaIndexLLMWrapper = DeprecationHelper(
    _LlamaIndexLLMWrapper,
    "LlamaIndexLLMWrapper is deprecated and will be removed in a future version. "
    "Use llm_factory instead: "
    "from openai import OpenAI; "
    "from ragas.llms import llm_factory; "
    "llm = llm_factory('gpt-4o-mini', client=OpenAI(api_key='...'))",
)

__all__ = [
    "BaseRagasLLM",
    "GroqLLMWrapper",
    "HaystackLLMWrapper",
    "InstructorBaseRagasLLM",
    "InstructorLLM",
    "LangchainLLMWrapper",
    "LlamaIndexLLMWrapper",
    "LiteLLMStructuredLLM",
    "OCIGenAIWrapper",
    "InstructorTypeVar",
    "llm_factory",
    "oci_genai_factory",
]
