import re
import typing as t
from dataclasses import dataclass

from ragas.testset.graph import Node
from ragas.testset.transforms.base import Extractor


@dataclass
class RegexBasedExtractor(Extractor):
    pattern: str = ""
    is_multiline: bool = False
    property_name: str = "regex"

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise ValueError(
                f"node.property('page_content') must be a string, found '{type(text)}'"
            )

        matches = re.findall(self.pattern, text, re.MULTILINE)
        return self.property_name, matches


# This regex pattern matches URLs, including those starting with "http://", "https://", or "www."
links_extractor_pattern = r"(?i)\b(?:https?://|www\.)\S+\b"
links_extractor = RegexBasedExtractor(
    pattern=links_extractor_pattern, is_multiline=True, property_name="links"
)

# This regex pattern matches emails, which typically follow the format "username@domain.extension".
# The alternation pairs the email match with a fallback that consumes a whole run of local-part
# characters. A long run that contains no "@" is therefore skipped in a single step instead of
# being re-scanned from every start position, which is what made a plain ``re.findall`` quadratic
# on such inputs. Only the captured email group (group 1) is kept; see ``EmailsExtractor`` below.
emails_extractor_pattern = (
    r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|[a-zA-Z0-9_.+-]+"
)


@dataclass
class EmailsExtractor(RegexBasedExtractor):
    """Extract emails from a node's ``page_content`` in linear time.

    ``pattern`` must expose the email as its first capturing group. Matches from
    the fallback alternative (runs of local-part characters skipped while
    scanning) have no group 1 and are dropped.
    """

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise ValueError(
                f"node.property('page_content') must be a string, found '{type(text)}'"
            )

        matches = [
            match.group(1)
            for match in re.finditer(self.pattern, text, re.MULTILINE)
            if match.group(1)
        ]
        return self.property_name, matches


emails_extractor = EmailsExtractor(
    pattern=emails_extractor_pattern, is_multiline=False, property_name="emails"
)

# This regex pattern matches Markdown headings, which start with a number sign (#) followed by a space,
# and the rest of the line is the heading text.
markdown_headings_pattern = r"^(#{1,6})\s+(.*)"
markdown_headings_extractor = RegexBasedExtractor(
    pattern=markdown_headings_pattern, is_multiline=True, property_name="headings"
)
