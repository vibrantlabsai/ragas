"""Tests for the griptape integration's optional-dependency handling."""

import importlib
import sys

import pytest


def test_missing_griptape_names_griptape_in_the_error(monkeypatch):
    """Without griptape installed, the ImportError must tell the user to
    install griptape, not another package."""
    # A None entry makes `import griptape...` raise ImportError, as it does
    # when the package is not installed.
    for name in ("griptape", "griptape.engines", "griptape.engines.rag"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "ragas.integrations.griptape", raising=False)

    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("ragas.integrations.griptape")

    message = str(exc_info.value)
    assert "pip install griptape" in message
    assert "opik" not in message.lower()
