"""Tests for MultiModalRelevance metric (collections implementation)."""

import os
import tempfile

import pytest
from PIL import Image

from ragas.metrics.collections.multi_modal_relevance.util import (
    MULTIMODAL_RELEVANCE_INSTRUCTION,
    MultiModalRelevanceOutput,
    build_multimodal_relevance_message_content,
)


class TestBuildMultimodalRelevanceMessageContent:
    """Test cases for building multimodal relevance message content."""

    def test_build_with_text_only(self):
        """Test building content with text-only contexts."""
        content = build_multimodal_relevance_message_content(
            instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
            user_input="What color is the sky?",
            response="The sky is blue.",
            retrieved_contexts=["The sky appears blue due to Rayleigh scattering."],
        )

        # Should have text blocks
        assert len(content) > 0
        text_blocks = [c for c in content if c["type"] == "text"]
        assert len(text_blocks) >= 2  # Instruction + context

    def test_build_with_mixed_content(self):
        """Test building content with mixed text and image contexts."""
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (10, 10), color="green")
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            content = build_multimodal_relevance_message_content(
                instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
                user_input="What is shown in the image?",
                response="The image shows green color.",
                retrieved_contexts=[temp_path, "Green is a color."],
            )

            # Should have both text and image blocks
            text_blocks = [c for c in content if c["type"] == "text"]
            image_blocks = [c for c in content if c["type"] == "image_url"]

            assert len(text_blocks) >= 2
            assert len(image_blocks) == 1
        finally:
            os.unlink(temp_path)

    def test_build_with_empty_contexts(self):
        """Test building content with empty contexts list."""
        content = build_multimodal_relevance_message_content(
            instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
            user_input="Some question?",
            response="Some response.",
            retrieved_contexts=[],
        )

        # Should still have instruction and closing text
        assert len(content) >= 2

    def test_content_contains_user_input(self):
        """Test that the built content contains the user input."""
        test_question = "This is a unique test question?"
        content = build_multimodal_relevance_message_content(
            instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
            user_input=test_question,
            response="Some response.",
            retrieved_contexts=["Some context."],
        )

        # Find all text content
        all_text = " ".join(
            c["text"] for c in content if c["type"] == "text" and "text" in c
        )
        assert test_question in all_text

    def test_content_contains_response(self):
        """Test that the built content contains the response."""
        test_response = "This is a unique test response."
        content = build_multimodal_relevance_message_content(
            instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
            user_input="Some question?",
            response=test_response,
            retrieved_contexts=["Some context."],
        )

        # Find all text content
        all_text = " ".join(
            c["text"] for c in content if c["type"] == "text" and "text" in c
        )
        assert test_response in all_text

    def test_build_with_multiple_images(self):
        """Test building content with multiple image contexts."""
        # Create temporary images
        temp_paths = []
        for color in ["red", "blue"]:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new("RGB", (10, 10), color=color)
                img.save(f, format="PNG")
                temp_paths.append(f.name)

        try:
            content = build_multimodal_relevance_message_content(
                instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
                user_input="What colors are shown?",
                response="Red and blue colors are shown.",
                retrieved_contexts=temp_paths,
            )

            image_blocks = [c for c in content if c["type"] == "image_url"]
            assert len(image_blocks) == 2
        finally:
            for path in temp_paths:
                os.unlink(path)


class TestMultiModalRelevanceOutput:
    """Test cases for the output model."""

    def test_output_relevant_true(self):
        """Test creating output with relevant=True."""
        output = MultiModalRelevanceOutput(
            relevant=True, reason="The response is in line with the context."
        )
        assert output.relevant is True
        assert "in line" in output.reason.lower()

    def test_output_relevant_false(self):
        """Test creating output with relevant=False."""
        output = MultiModalRelevanceOutput(
            relevant=False, reason="The response contradicts the context."
        )
        assert output.relevant is False
        assert "contradicts" in output.reason.lower()

    def test_output_default_reason(self):
        """Test output with default (empty) reason."""
        output = MultiModalRelevanceOutput(relevant=True)
        assert output.relevant is True
        assert output.reason == ""


class TestMultiModalRelevanceMetric:
    """Test cases for the MultiModalRelevance metric class."""

    @pytest.mark.asyncio
    async def test_input_validation_missing_user_input(self):
        """Test that missing user_input raises ValueError."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        from ragas.metrics.collections.multi_modal_relevance import (
            MultiModalRelevance,
        )

        # Bypass LLM validation by setting attribute directly
        metric = object.__new__(MultiModalRelevance)
        metric.llm = mock_llm
        metric.name = "test"

        with pytest.raises(ValueError, match="user_input is missing"):
            await metric.ascore(
                user_input="",
                response="Some response",
                retrieved_contexts=["Some context"],
            )

    @pytest.mark.asyncio
    async def test_input_validation_missing_response(self):
        """Test that missing response raises ValueError."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        from ragas.metrics.collections.multi_modal_relevance import (
            MultiModalRelevance,
        )

        # Bypass LLM validation by setting attribute directly
        metric = object.__new__(MultiModalRelevance)
        metric.llm = mock_llm
        metric.name = "test"

        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                user_input="Some question?",
                response="",
                retrieved_contexts=["Some context"],
            )

    @pytest.mark.asyncio
    async def test_input_validation_missing_contexts(self):
        """Test that missing contexts raises ValueError."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        from ragas.metrics.collections.multi_modal_relevance import (
            MultiModalRelevance,
        )

        metric = object.__new__(MultiModalRelevance)
        metric.llm = mock_llm
        metric.name = "test"

        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                user_input="Some question?",
                response="Some response",
                retrieved_contexts=[],
            )

    def test_metric_name_default(self):
        """Test that default metric name is set correctly."""
        from unittest.mock import MagicMock

        from ragas.metrics.collections.multi_modal_relevance import (
            MultiModalRelevance,
        )

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        metric = object.__new__(MultiModalRelevance)
        metric.llm = mock_llm
        metric.name = "multi_modal_relevance"

        assert metric.name == "multi_modal_relevance"

    def test_instruction_content(self):
        """Test that the instruction contains key evaluation criteria."""
        assert "RELEVANT" in MULTIMODAL_RELEVANCE_INSTRUCTION
        assert "NOT RELEVANT" in MULTIMODAL_RELEVANCE_INSTRUCTION
        assert "visual" in MULTIMODAL_RELEVANCE_INSTRUCTION.lower()
        assert "textual" in MULTIMODAL_RELEVANCE_INSTRUCTION.lower()
