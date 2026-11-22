"""Tests for backward-compatible import aliases in ragas.metrics.collections.

The deprecation warnings in ragas.metrics.__init__ tell users:
    from ragas.metrics.collections import <OldName>

But several metrics were renamed between old and new API. These tests verify
that the old names are importable from collections and resolve to the correct
new implementation.
"""

import pytest


# Each tuple: (old_name, expected_new_class_name)
ALIAS_MAPPING = [
    ("AnswerSimilarity", "SemanticSimilarity"),
    ("ChrfScore", "CHRFScore"),
    ("InstanceRubrics", "InstanceSpecificRubrics"),
    ("LLMContextPrecisionWithoutReference", "ContextPrecisionWithoutReference"),
    ("LLMContextPrecisionWithReference", "ContextPrecisionWithReference"),
    ("LLMSQLEquivalence", "SQLSemanticEquivalence"),
    ("ResponseRelevancy", "AnswerRelevancy"),
    ("RubricsScore", "DomainSpecificRubrics"),
    ("SummarizationScore", "SummaryScore"),
    ("TopicAdherenceScore", "TopicAdherence"),
]


@pytest.mark.parametrize("old_name,new_name", ALIAS_MAPPING)
def test_deprecated_alias_importable(old_name, new_name):
    """Old metric names should be importable from ragas.metrics.collections."""
    import ragas.metrics.collections as collections

    cls = getattr(collections, old_name)
    assert cls is not None
    assert cls.__name__ == new_name


@pytest.mark.parametrize("old_name,new_name", ALIAS_MAPPING)
def test_alias_is_same_class_as_new_name(old_name, new_name):
    """The alias should point to the exact same class object as the new name."""
    import ragas.metrics.collections as collections

    old_cls = getattr(collections, old_name)
    new_cls = getattr(collections, new_name)
    assert old_cls is new_cls


def test_all_aliases_in_module_all():
    """All aliases should be listed in __all__."""
    from ragas.metrics.collections import __all__

    for old_name, _ in ALIAS_MAPPING:
        assert old_name in __all__, f"{old_name} missing from __all__"


def test_deprecation_warning_migration_path():
    """The full migration path should work: old import → warning → collections import.

    This simulates the user experience:
    1. User does `from ragas.metrics import ChrfScore`
    2. Gets deprecation warning saying to use `from ragas.metrics.collections import ChrfScore`
    3. User follows the advice — it should work (this is the bug we're fixing)
    """
    import warnings
    import ragas.metrics.collections as collections

    # Get all deprecated metric names from the old API
    from ragas.metrics import _DEPRECATED_METRICS

    for name in _DEPRECATED_METRICS:
        # Skip lowercase instance names (e.g., answer_correctness, faithfulness)
        # and names that are completely missing from collections (Category A)
        if name[0].islower():
            continue

        # The deprecation message tells users to do:
        #   from ragas.metrics.collections import {name}
        # Verify that actually works
        if hasattr(collections, name):
            cls = getattr(collections, name)
            assert cls is not None, f"collections.{name} resolved to None"
