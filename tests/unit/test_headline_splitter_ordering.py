import pytest

from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms.splitters import HeadlineSplitter


@pytest.mark.asyncio
async def test_headline_splitter_handles_repeated_headline_text():
    """Headlines are located at their actual sequential positions.

    When the text of a later headline also appears earlier in the document
    (for example a section that is mentioned before it actually starts), the
    splitter must not match that earlier mention. Doing so produces
    out-of-order section boundaries, which in turn yields empty chunks and
    chunks whose content comes from the wrong section.
    """
    intro = (
        "Introduction\n"
        + ("This is the intro. " * 30)
        # "Conclusion" is mentioned here, before the actual Conclusion section.
        + "We will cover the Conclusion later. "
        + ("More intro text here. " * 20)
    )
    methods = "Methods\n" + ("We describe the methods in detail. " * 40)
    conclusion = "Conclusion\n" + ("We conclude the work here. " * 40)
    text = intro + methods + conclusion
    headlines = ["Introduction", "Methods", "Conclusion"]

    node = Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": text, "headlines": headlines},
    )
    splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
    nodes, _ = await splitter.split(node)

    contents = [n.get_property("page_content") for n in nodes]

    # No empty chunks should be produced.
    assert all(content for content in contents)

    # The "Methods" section content must be preserved and not dropped or
    # mangled by an out-of-order boundary.
    assert any("We describe the methods in detail." in content for content in contents)

    # The chunk that begins the Conclusion section must start at the real
    # heading, not at the earlier mention inside the introduction.
    assert any(content.startswith("Conclusion\n") for content in contents)
