"""Tests for the Griptape integration."""

from __future__ import annotations

import builtins
import sys

import pytest


def test_missing_griptape_import_error(monkeypatch):
    """Importing the Griptape integration without griptape installed should
    raise an ImportError that points to the griptape package, not opik."""
    real_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name.startswith("griptape"):
            raise ImportError("No module named 'griptape'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
    monkeypatch.delitem(sys.modules, "ragas.integrations.griptape", raising=False)

    with pytest.raises(ImportError) as exc_info:
        import ragas.integrations.griptape  # noqa: F401

    message = str(exc_info.value)
    assert "griptape" in message.lower()
    assert "opik" not in message.lower()
    # the original ImportError for the missing griptape module is chained
    assert isinstance(exc_info.value.__cause__, ImportError)
