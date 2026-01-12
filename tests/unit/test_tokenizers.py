"""Tests for ragas.tokenizers module."""

from __future__ import annotations

import socket


def test_tokenizer_import_without_network(monkeypatch):
    """Import should work without network (for offline environments)."""

    def block_network(*args, **kwargs):
        raise OSError("Network blocked for testing")

    monkeypatch.setattr(socket, "getaddrinfo", block_network)

    from ragas.tokenizers import DEFAULT_TOKENIZER, get_default_tokenizer

    assert DEFAULT_TOKENIZER is not None
    assert get_default_tokenizer is not None


def test_default_tokenizer_encode_decode():
    from ragas.tokenizers import DEFAULT_TOKENIZER

    text = "Hello world"
    tokens = DEFAULT_TOKENIZER.encode(text)
    decoded = DEFAULT_TOKENIZER.decode(tokens)

    assert len(tokens) > 0
    assert decoded == text


def test_get_default_tokenizer_singleton():
    from ragas.tokenizers import get_default_tokenizer

    t1 = get_default_tokenizer()
    t2 = get_default_tokenizer()

    assert t1 is t2


def test_default_tokenizer_with_dataclass():
    """Ensure backwards compat with existing default_factory usage."""
    from dataclasses import dataclass, field

    from ragas.tokenizers import DEFAULT_TOKENIZER, BaseTokenizer

    @dataclass
    class TestClass:
        tokenizer: BaseTokenizer = field(default_factory=lambda: DEFAULT_TOKENIZER)

    obj = TestClass()
    assert len(obj.tokenizer.encode("test")) > 0
