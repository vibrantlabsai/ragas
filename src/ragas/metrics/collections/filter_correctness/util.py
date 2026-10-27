"""Helpers for structural retrieval-filter checks."""

from __future__ import annotations

import typing as t


def normalize_allowed_values(allowed_values: t.Any) -> t.FrozenSet[t.Any]:
    """Normalize equality / ``$in`` style allowed values to a frozenset.

    Accepts:
    - a single scalar (``str``, ``int``, …) for equality filters
    - a non-empty sequence for membership / ``$in`` filters
    """
    if isinstance(allowed_values, (bytes, bytearray)):
        raise TypeError(
            "allowed_values must be a hashable scalar or a sequence of hashable values"
        )
    if isinstance(allowed_values, str):
        return frozenset({allowed_values})
    if isinstance(allowed_values, t.Sequence):
        if len(allowed_values) == 0:
            raise ValueError("allowed_values must not be empty")
        try:
            return frozenset(allowed_values)
        except TypeError as exc:
            raise TypeError(
                "allowed_values must be a hashable scalar or a sequence of hashable values"
            ) from exc
    try:
        return frozenset({allowed_values})
    except TypeError as exc:
        raise TypeError(
            "allowed_values must be a hashable scalar or a sequence of hashable values"
        ) from exc


def value_respects_filter(value: t.Any, allowed: t.FrozenSet[t.Any]) -> bool:
    """Return True if ``value`` is allowed by equality / membership semantics.

    - Scalar values must be members of ``allowed``.
    - List/tuple values must be a non-empty subset of ``allowed`` (every element allowed).
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return False
        return all(item in allowed for item in value)
    return value in allowed


def inspect_retrieved_metadatas(
    retrieved_metadatas: t.Sequence[t.Mapping[str, t.Any]],
    filter_field: str,
    allowed: t.FrozenSet[t.Any],
) -> t.Dict[str, t.Any]:
    """Inspect retrieved chunk metadata against a field filter.

    Returns counts and offending values. Chunks missing ``filter_field`` are
    counted separately and are not treated as filter violations.
    """
    missing_metadata = 0
    violations = 0
    checked = 0
    offending: t.List[t.Any] = []
    seen_offending: t.Set[t.Any] = set()

    for meta in retrieved_metadatas:
        if not isinstance(meta, t.Mapping):
            raise TypeError(
                "each item in retrieved_metadatas must be a mapping of chunk metadata"
            )
        if filter_field not in meta or meta[filter_field] is None:
            missing_metadata += 1
            continue

        checked += 1
        value = meta[filter_field]
        if value_respects_filter(value, allowed):
            continue

        violations += 1
        # Record a stable, human-readable offending token for the reason string.
        if isinstance(value, (list, tuple)):
            for item in value:
                if item not in allowed and item not in seen_offending:
                    seen_offending.add(item)
                    offending.append(item)
        elif value not in seen_offending:
            seen_offending.add(value)
            offending.append(value)

    return {
        "checked": checked,
        "violations": violations,
        "missing_metadata": missing_metadata,
        "offending_values": offending,
    }
