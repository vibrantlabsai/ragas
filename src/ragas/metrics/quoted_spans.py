"""
Quoted Spans Alignment Metric
================================

This module provides a simple metric to measure citation alignment for quoted spans
in model-generated answers. The idea is to compute the fraction of quoted spans
appearing verbatim in any of the provided source passages.  If an answer quotes
facts that cannot be found in the sources, the metric will reflect that drift.

The metric function is designed to be plug‑and‑play in existing evaluation
pipelines.  It returns a score in the range [0, 1] along with the raw counts for
matched and total quoted spans.  It performs light normalization by collapsing
whitespace and lower‑casing strings.  You can adjust the minimum length of a
quoted span and choose to disable case folding if desired.
"""

from __future__ import annotations

import re
from typing import Dict, Sequence

# Regular expression to extract both straight and curly quoted spans.  Matches
# pairs of quotes and captures the inner text.  Also supports Arabic/French
# guillemets (« ») commonly used in Arabic and other non-English texts.
_QUOTE_RE = re.compile(r"[\"" "''`´«](.*?)[\"" "''`´»]")


def _normalize(text: str) -> str:
    """Normalize text by collapsing whitespace and lower‑casing it."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_quoted_spans(answer: str, *, min_len: int = 3) -> Sequence[str]:
    """
    Extract quoted spans from an answer.

    Parameters
    ----------
    answer: str
        The model answer to search for quoted spans.
    min_len: int, optional
        Minimum number of words required for a span to be considered.  Shorter
        spans are ignored to avoid spurious matches.

    Returns
    -------
    Sequence[str]
        A list of quoted spans (strings) that meet the minimum length
        requirement.
    """
    spans: list[str] = []
    for match in _QUOTE_RE.finditer(answer):
        span = (match.group(1) or "").strip()
        # filter out spans shorter than min_len words
        if len(span.split()) >= min_len:
            spans.append(span)
    return spans


def quoted_spans_alignment(
    answers: Sequence[str],
    sources: Sequence[Sequence[str]],
    *,
    casefold: bool = True,
    min_len: int = 3,
) -> Dict[str, float]:
    """
    Compute the citation alignment score for quoted spans in model answers.

    Parameters
    ----------
    answers: Sequence[str]
        List of model answers (length N).
    sources: Sequence[Sequence[str]]
        List of lists (length N) containing passages for each answer.
    casefold: bool, optional
        Whether to normalize text by lower‑casing before matching.  Defaults
        to True.
    min_len: int, optional
        Minimum number of words in a quoted span.  Defaults to 3.

    Returns
    -------
    Dict[str, float]
        A dictionary containing:
            - "citation_alignment_quoted_spans": the fraction of quoted
              spans found verbatim in the provided sources.
            - "matched": number of spans that were matched
            - "total": total number of spans considered

    Notes
    -----
    If no quoted spans are found across the dataset, the score is defined as
    0.0, with matched=0 and total=0.  Matching is substring matching on
    normalized text.
    """
    if len(answers) != len(sources):
        raise ValueError("answers and sources must have the same length")
    matched = 0
    total = 0

    for answer, src_list in zip(answers, sources):
        spans = _extract_quoted_spans(answer, min_len=min_len)
        if not spans:
            continue
        # join all sources for this answer into one string
        joined_sources = " ".join(src_list)
        if casefold:
            normalized_sources = _normalize(joined_sources)
        else:
            normalized_sources = joined_sources

        for span in spans:
            total += 1
            span_norm = _normalize(span) if casefold else span
            # check if the normalized span appears in the normalized sources
            if span_norm and span_norm in normalized_sources:
                matched += 1

    score = (matched / total) if total else 0.0
    return {
        "citation_alignment_quoted_spans": float(score),
        "matched": float(matched),
        "total": float(total),
    }
