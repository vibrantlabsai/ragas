import typing as t
from dataclasses import dataclass

from ragas.testset.graph import Node, NodeType, Relationship
from ragas.testset.transforms.base import Splitter
from ragas.utils import num_tokens_from_string


@dataclass
class HeadlineSplitter(Splitter):
    min_tokens: int = 300
    max_tokens: int = 1000

    def adjust_chunks(self, chunks):
        adjusted_chunks = []
        current_chunk = ""

        for chunk in chunks:
            chunk_token_count = num_tokens_from_string(chunk)

            # Split chunks that are over max_tokens
            while chunk_token_count > self.max_tokens:
                # For chunks over max_tokens, we need to split by words since we can't
                # easily split tokens without losing token boundary information
                words = chunk.split()
                # Estimate split point based on token ratio
                split_ratio = self.max_tokens / chunk_token_count
                split_point = max(1, int(len(words) * split_ratio))

                chunk_part = " ".join(words[:split_point])
                adjusted_chunks.append(chunk_part)

                # Continue with remaining part
                chunk = " ".join(words[split_point:])
                chunk_token_count = num_tokens_from_string(chunk)

            # Handle chunks that are under min_tokens
            if chunk_token_count < self.min_tokens:
                if current_chunk:
                    current_chunk += " " + chunk
                    if num_tokens_from_string(current_chunk) >= self.min_tokens:
                        adjusted_chunks.append(current_chunk)
                        current_chunk = ""
                else:
                    current_chunk = chunk
            else:
                if current_chunk:
                    adjusted_chunks.append(current_chunk)
                    current_chunk = ""
                adjusted_chunks.append(chunk)

        # Append any remaining chunk
        if current_chunk:
            adjusted_chunks.append(current_chunk)

        return adjusted_chunks

    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        if text is None:
            raise ValueError("'page_content' property not found in this node")

        headlines = node.get_property("headlines")
        if headlines is None:
            raise ValueError("'headlines' property not found in this node")

        if num_tokens_from_string(text) < self.min_tokens:
            return [node], []
        # create the chunks for the different sections
        indices = [0]
        for headline in headlines:
            index = text.find(headline)
            if index != -1:
                indices.append(index)
        indices.append(len(text))
        chunks = [text[start:end] for start, end in zip(indices, indices[1:])]
        chunks = self.adjust_chunks(chunks)

        # if there was no headline, return the original node
        if len(chunks) == 1:
            return [node], []

        # create the nodes
        nodes = [
            Node(type=NodeType.CHUNK, properties={"page_content": chunk})
            for chunk in chunks
        ]

        # create the relationships for children
        relationships = []
        for child_node in nodes:
            relationships.append(
                Relationship(
                    type="child",
                    source=node,
                    target=child_node,
                )
            )

        # create the relationships for the next nodes
        for i, child_node in enumerate(nodes):
            if i < len(nodes) - 1:
                relationships.append(
                    Relationship(
                        type="next",
                        source=child_node,
                        target=nodes[i + 1],
                    )
                )
        return nodes, relationships
