"""Citation faithfulness prompt classes, models, and pure helpers."""

from __future__ import annotations

import re
import typing as t

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt

# --------------------------------------------------------------------------- #
# Pure helpers (no LLM) - unit-testable in isolation, like quoted_spans/util.py
# --------------------------------------------------------------------------- #

# Matches an inline citation marker such as ``[1]``, ``[ 12 ]`` or ``[1, 2]``.
# The captured group is the comma-separated list of 1-based passage indices.
_CITATION_MARKER_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def has_citation_markers(text: str) -> bool:
    """
    Return True if the text contains at least one ``[N]`` citation marker.

    Used as a fast pre-check for the degenerate no-citation case before any LLM
    call is made. Markers may be single (``[1]``), spaced (``[ 12 ]``), or
    comma-grouped (``[1, 2]``).

    Args:
        text: The response text to scan.

    Returns:
        True if a citation marker is present, False otherwise.
    """
    if not text:
        return False
    return _CITATION_MARKER_RE.search(text) is not None


def parse_citation_markers(text: str) -> t.List[int]:
    """
    Extract every 1-based passage index from the ``[N]`` markers in the text.

    A grouped marker such as ``[1, 2]`` yields two indices. Indices are returned
    in the order they appear.

    Args:
        text: The response text to parse.

    Returns:
        A list of 1-based passage indices in appearance order.
    """
    if not text:
        return []
    indices: t.List[int] = []
    for match in _CITATION_MARKER_RE.finditer(text):
        for part in match.group(1).split(","):
            indices.append(int(part.strip()))
    return indices


def compute_citation_score(
    claim_citations: t.Sequence[t.Sequence[int]],
    num_contexts: int,
    verdicts: t.Mapping[t.Tuple[int, int], bool],
) -> float:
    """
    Compute the citation-attribution faithfulness score.

    The score is the fraction of cited claims whose citations are ALL correct. A
    single citation is correct only if it is in range (a 1-based index of a
    retrieved passage) and the ``verdicts`` lookup marks it as supporting the
    claim. An out-of-range citation is always incorrect and is never expected to
    appear in ``verdicts``.

    Args:
        claim_citations: One entry per cited claim, each the list of 1-based
            passage indices cited for that claim. Each entry is expected to carry
            at least one citation.
        num_contexts: The number of retrieved passages (defines the valid
            1..num_contexts index range).
        verdicts: Lookup of (claim_index, citation_index) -> whether the cited
            passage supports that specific claim.

    Returns:
        The fraction of cited claims that are fully correctly attributed, or
        ``float("nan")`` when there are no cited claims.
    """
    if not claim_citations:
        return float("nan")

    correct = 0
    for claim_idx, citations in enumerate(claim_citations):
        if not citations:
            # A claim with no citations is not a cited claim; treat as incorrect.
            continue
        claim_ok = True
        for citation in citations:
            in_range = 1 <= citation <= num_contexts
            if not in_range or not verdicts.get((claim_idx, citation), False):
                claim_ok = False
                break
        if claim_ok:
            correct += 1

    return correct / len(claim_citations)


# --------------------------------------------------------------------------- #
# Step 1: cited-claim extraction
# --------------------------------------------------------------------------- #


class CitationClaimInput(BaseModel):
    """Input model for cited-claim extraction."""

    question: str = Field(..., description="The question being answered")
    answer: str = Field(
        ...,
        description=(
            "The answer text, containing inline citation markers of the form [N] "
            "where N is the 1-based index of a retrieved passage."
        ),
    )


class CitedClaim(BaseModel):
    """A single extracted claim together with the passages it cites."""

    claim: str = Field(
        ..., description="A single, self-contained claim extracted from the answer."
    )
    citations: t.List[int] = Field(
        ...,
        description=(
            "The 1-based passage indices cited for this claim, taken from the [N] "
            "markers attached to it. Empty if the claim carries no citation."
        ),
    )


class CitationClaimOutput(BaseModel):
    """Structured output for cited-claim extraction."""

    claims: t.List[CitedClaim] = Field(
        ..., description="The cited claims extracted from the answer."
    )


class CitationClaimPrompt(BasePrompt[CitationClaimInput, CitationClaimOutput]):
    """Prompt for breaking an answer into claims with their cited passage indices."""

    input_model = CitationClaimInput
    output_model = CitationClaimOutput

    instruction = """Given a question and an answer that contains inline citation markers of the form [N] (where N is the 1-based index of a retrieved passage), break the answer into one or more fully understandable claims. Ensure that no pronouns are used in any claim. For each claim, record the list of passage indices that are cited for it, taken from the [N] markers attached to that claim. Only include claims that carry at least one citation marker."""

    examples = [
        (
            CitationClaimInput(
                question="What are some properties of green tea?",
                answer=(
                    "Green tea contains antioxidants called catechins [1]. "
                    "Green tea may also improve brain function [2]."
                ),
            ),
            CitationClaimOutput(
                claims=[
                    CitedClaim(
                        claim="Green tea contains antioxidants called catechins.",
                        citations=[1],
                    ),
                    CitedClaim(
                        claim="Green tea may improve brain function.",
                        citations=[2],
                    ),
                ]
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Step 2: per-citation attribution verification
# --------------------------------------------------------------------------- #


class CitationVerdictItem(BaseModel):
    """A single (claim, citation, cited-passage) triple to be judged."""

    claim: str = Field(..., description="the claim, word-by-word")
    citation: int = Field(..., description="the 1-based index of the cited passage")
    context: str = Field(..., description="the text of the cited passage")


class CitationVerdictInput(BaseModel):
    """Input model for citation-attribution verification."""

    items: t.List[CitationVerdictItem] = Field(
        ..., description="The (claim, citation, cited-passage) triples to judge"
    )


class CitationAttributionVerdict(BaseModel):
    """The attribution verdict for a single cited claim."""

    claim: str = Field(..., description="the original claim, word-by-word")
    citation: int = Field(..., description="the 1-based index of the cited passage")
    reason: str = Field(..., description="the reason for the verdict")
    verdict: int = Field(
        ...,
        description=(
            "1 if the cited passage supports the claim, 0 if the cited passage "
            "does not support the claim."
        ),
    )


class CitationVerdictOutput(BaseModel):
    """Structured output for citation-attribution verification."""

    verdicts: t.List[CitationAttributionVerdict] = Field(
        ..., description="One verdict per input item, in the same order."
    )


class CitationAttributionPrompt(
    BasePrompt[CitationVerdictInput, CitationVerdictOutput]
):
    """Prompt for verifying that each cited passage supports the claim it is attached to."""

    input_model = CitationVerdictInput
    output_model = CitationVerdictOutput

    instruction = """Your task is to judge, for each item, whether the SPECIFIC cited passage supports the claim it is attached to. For each item return verdict 1 if the claim can be directly inferred from that item's cited passage, or 0 if it cannot. Judge only the passage given in the item. Do not credit the citation because the claim might be supported by some other passage: if the cited passage itself does not support the claim, the verdict is 0. Return one verdict per input item, in the same order."""

    examples = [
        (
            CitationVerdictInput(
                items=[
                    CitationVerdictItem(
                        claim="Green tea contains antioxidants called catechins.",
                        citation=1,
                        context=(
                            "Green tea is rich in catechins, a type of natural "
                            "antioxidant."
                        ),
                    ),
                    CitationVerdictItem(
                        claim="Green tea may improve brain function.",
                        citation=2,
                        context=(
                            "Green tea has a pleasant aroma and is widely consumed "
                            "across East Asia."
                        ),
                    ),
                ]
            ),
            CitationVerdictOutput(
                verdicts=[
                    CitationAttributionVerdict(
                        claim="Green tea contains antioxidants called catechins.",
                        citation=1,
                        reason=(
                            "The cited passage states green tea is rich in catechins, "
                            "a type of antioxidant, which directly supports the claim."
                        ),
                        verdict=1,
                    ),
                    CitationAttributionVerdict(
                        claim="Green tea may improve brain function.",
                        citation=2,
                        reason=(
                            "The cited passage only mentions the aroma and consumption "
                            "of green tea. It says nothing about brain function, so it "
                            "does not support the claim."
                        ),
                        verdict=0,
                    ),
                ]
            ),
        ),
    ]
