"""
Tokenizer abstractions for Ragas.

This module provides a unified interface for different tokenizer implementations,
supporting both tiktoken (OpenAI) and HuggingFace tokenizers.
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import tiktoken


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> t.List[int]:
        """Encode text into token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: t.List[int]) -> str:
        """Decode token IDs back into text."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.encode(text))


class TiktokenWrapper(BaseTokenizer):
    """Wrapper for tiktoken encodings (OpenAI tokenizers)."""

    def __init__(
        self,
        encoding: t.Optional[tiktoken.Encoding] = None,
        model_name: t.Optional[str] = None,
        encoding_name: t.Optional[str] = None,
    ):
        """
        Initialize TiktokenWrapper.

        Parameters
        ----------
        encoding : tiktoken.Encoding, optional
            A pre-initialized tiktoken encoding.
        model_name : str, optional
            Model name to get encoding for (e.g., "gpt-4", "gpt-3.5-turbo").
        encoding_name : str, optional
            Encoding name (e.g., "cl100k_base", "o200k_base").

        If none provided, defaults to "o200k_base" encoding.
        """
        if encoding is not None:
            self._encoding = encoding
        elif model_name is not None:
            self._encoding = tiktoken.encoding_for_model(model_name)
        elif encoding_name is not None:
            self._encoding = tiktoken.get_encoding(encoding_name)
        else:
            self._encoding = tiktoken.get_encoding("o200k_base")

    def encode(self, text: str) -> t.List[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: t.List[int]) -> str:
        return self._encoding.decode(tokens)

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Access the underlying tiktoken encoding."""
        return self._encoding


class HuggingFaceTokenizer(BaseTokenizer):
    """Wrapper for HuggingFace tokenizers."""

    def __init__(
        self,
        tokenizer: t.Optional[t.Any] = None,
        model_name: t.Optional[str] = None,
    ):
        """
        Initialize HuggingFaceTokenizer.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer or PreTrainedTokenizerFast, optional
            A pre-initialized HuggingFace tokenizer.
        model_name : str, optional
            Model name or path to load tokenizer from (e.g., "meta-llama/Llama-2-7b").

        One of tokenizer or model_name must be provided.
        """
        if tokenizer is not None:
            self._tokenizer = tokenizer
        elif model_name is not None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers package is required for HuggingFace tokenizers. "
                    "Install it with: pip install transformers"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("Either tokenizer or model_name must be provided")

    def encode(self, text: str) -> t.List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: t.List[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    @property
    def tokenizer(self) -> t.Any:
        """Access the underlying HuggingFace tokenizer."""
        return self._tokenizer


# Default tokenizer instance
DEFAULT_TOKENIZER = TiktokenWrapper(encoding_name="o200k_base")


def get_tokenizer(
    tokenizer_type: str = "tiktoken",
    model_name: t.Optional[str] = None,
    encoding_name: t.Optional[str] = None,
) -> BaseTokenizer:
    """
    Factory function to get a tokenizer instance.

    Parameters
    ----------
    tokenizer_type : str
        Type of tokenizer: "tiktoken" or "huggingface".
    model_name : str, optional
        Model name for the tokenizer.
    encoding_name : str, optional
        Encoding name (only for tiktoken).

    Returns
    -------
    BaseTokenizer
        A tokenizer instance.

    Examples
    --------
    >>> # Get default tiktoken tokenizer
    >>> tokenizer = get_tokenizer()

    >>> # Get tiktoken for a specific model
    >>> tokenizer = get_tokenizer("tiktoken", model_name="gpt-4")

    >>> # Get HuggingFace tokenizer
    >>> tokenizer = get_tokenizer("huggingface", model_name="meta-llama/Llama-2-7b")
    """
    if tokenizer_type == "tiktoken":
        return TiktokenWrapper(model_name=model_name, encoding_name=encoding_name)
    elif tokenizer_type == "huggingface":
        if model_name is None:
            raise ValueError("model_name is required for HuggingFace tokenizers")
        return HuggingFaceTokenizer(model_name=model_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
