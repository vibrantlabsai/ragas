import asyncio
import time

from ragas.testset.graph import Node
from ragas.testset.transforms.extractors.regex_based import emails_extractor


def _emails(text: str) -> list:
    node = Node(properties={"page_content": text})
    key, value = asyncio.run(emails_extractor.extract(node))
    assert key == "emails"
    return value


def test_extracts_simple_emails():
    assert _emails("contact a.b+x@mail.co, x@y.com here") == [
        "a.b+x@mail.co",
        "x@y.com",
    ]


def test_extracts_adjacent_emails():
    # The second email starts immediately after the first match, inside a run of
    # local-part characters. Both must still be returned.
    assert _emails("a@b.co+x@y.com") == ["a@b.co", "+x@y.com"]


def test_no_emails_returns_empty():
    assert _emails("no address here, just some words.") == []


def test_long_run_without_at_is_not_quadratic():
    # A long run of local-part characters with no "@" previously took several
    # seconds (quadratic in the length of the run). It must now finish quickly.
    text = "a" * 200_000
    start = time.perf_counter()
    result = _emails(text)
    elapsed = time.perf_counter() - start
    assert result == []
    assert elapsed < 2.0, (
        f"emails extraction took {elapsed:.2f}s on 200k chars — possible "
        "quadratic-time regression"
    )
