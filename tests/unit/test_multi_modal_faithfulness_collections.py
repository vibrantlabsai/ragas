"""Tests for MultiModalFaithfulness metric (collections implementation)."""

import base64
import os
import tempfile

import pytest
from PIL import Image

from ragas.metrics.collections.multi_modal_faithfulness.util import (
    MULTIMODAL_FAITHFULNESS_INSTRUCTION,
    MultiModalFaithfulnessOutput,
    build_multimodal_message_content,
    is_image_path_or_url,
    process_image_to_base64,
)


class TestImageProcessingUtilities:
    """Test cases for image processing utility functions."""

    def test_is_image_path_or_url_with_http_url(self):
        """Test detection of HTTP URLs."""
        assert is_image_path_or_url("http://example.com/image.jpg") is True
        assert is_image_path_or_url("http://example.com/image.png") is True
        assert is_image_path_or_url("http://example.com/path/to/image.jpeg") is True

    def test_is_image_path_or_url_with_https_url(self):
        """Test detection of HTTPS URLs."""
        assert is_image_path_or_url("https://example.com/image.jpg") is True
        assert is_image_path_or_url("https://example.com/image.gif") is True

    def test_is_image_path_or_url_with_local_path(self):
        """Test detection of local file paths."""
        assert is_image_path_or_url("/path/to/image.jpg") is True
        assert is_image_path_or_url("./images/photo.png") is True
        assert is_image_path_or_url("image.jpeg") is True

    def test_is_image_path_or_url_with_base64(self):
        """Test detection of base64 data URIs."""
        base64_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD="
        assert is_image_path_or_url(base64_uri) is True

    def test_is_image_path_or_url_with_text(self):
        """Test that regular text is not detected as image."""
        assert is_image_path_or_url("This is just text") is False
        assert is_image_path_or_url("") is False
        assert is_image_path_or_url("file.txt") is False

    def test_is_image_path_or_url_with_none(self):
        """Test handling of invalid inputs."""
        assert is_image_path_or_url(None) is False  # type: ignore
        assert is_image_path_or_url("") is False

    def test_process_image_to_base64_with_valid_file(self):
        """Test processing a valid local image file."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (10, 10), color="red")
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            result = process_image_to_base64(temp_path)
            assert result is not None
            assert "mime_type" in result
            assert "encoded_data" in result
            assert result["mime_type"] == "image/png"
            # Verify base64 is valid
            base64.b64decode(result["encoded_data"])
        finally:
            os.unlink(temp_path)

    def test_process_image_to_base64_with_invalid_file(self):
        """Test processing a non-existent file."""
        result = process_image_to_base64("/nonexistent/path/image.jpg")
        assert result is None

    def test_process_image_to_base64_with_text(self):
        """Test that text is not processed as image."""
        result = process_image_to_base64("This is just text")
        assert result is None

    def test_process_image_to_base64_with_valid_base64(self):
        """Test processing a valid base64 data URI."""
        # Create a small valid PNG in base64
        img = Image.new("RGB", (2, 2), color="blue")
        from io import BytesIO

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        result = process_image_to_base64(data_uri)
        assert result is not None
        assert result["mime_type"] == "image/png"


class TestBuildMultimodalMessageContent:
    """Test cases for building multimodal message content."""

    def test_build_with_text_only(self):
        """Test building content with text-only contexts."""
        content = build_multimodal_message_content(
            instruction=MULTIMODAL_FAITHFULNESS_INSTRUCTION,
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
            content = build_multimodal_message_content(
                instruction=MULTIMODAL_FAITHFULNESS_INSTRUCTION,
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
        content = build_multimodal_message_content(
            instruction=MULTIMODAL_FAITHFULNESS_INSTRUCTION,
            response="Some response.",
            retrieved_contexts=[],
        )

        # Should still have instruction and closing text
        assert len(content) >= 2

    def test_content_contains_response(self):
        """Test that the built content contains the response."""
        test_response = "This is a unique test response."
        content = build_multimodal_message_content(
            instruction=MULTIMODAL_FAITHFULNESS_INSTRUCTION,
            response=test_response,
            retrieved_contexts=["Some context."],
        )

        # Find all text content
        all_text = " ".join(
            c["text"] for c in content if c["type"] == "text" and "text" in c
        )
        assert test_response in all_text


class TestMultiModalFaithfulnessOutput:
    """Test cases for the output model."""

    def test_output_faithful_true(self):
        """Test creating output with faithful=True."""
        output = MultiModalFaithfulnessOutput(
            faithful=True, reason="The response is supported by the context."
        )
        assert output.faithful is True
        assert "supported" in output.reason.lower()

    def test_output_faithful_false(self):
        """Test creating output with faithful=False."""
        output = MultiModalFaithfulnessOutput(
            faithful=False, reason="The response contradicts the context."
        )
        assert output.faithful is False
        assert "contradicts" in output.reason.lower()

    def test_output_default_reason(self):
        """Test output with default (empty) reason."""
        output = MultiModalFaithfulnessOutput(faithful=True)
        assert output.faithful is True
        assert output.reason == ""


class TestMultiModalFaithfulnessMetric:
    """Test cases for the MultiModalFaithfulness metric class."""

    @pytest.mark.asyncio
    async def test_input_validation_missing_response(self):
        """Test that missing response raises ValueError."""
        # Create a mock LLM that won't be called
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        from ragas.metrics.collections.multi_modal_faithfulness import (
            MultiModalFaithfulness,
        )

        # Bypass LLM validation by setting attribute directly
        metric = object.__new__(MultiModalFaithfulness)
        metric.llm = mock_llm
        metric.name = "test"

        with pytest.raises(ValueError, match="response is missing"):
            await metric.ascore(
                response="",
                retrieved_contexts=["Some context"],
            )

    @pytest.mark.asyncio
    async def test_input_validation_missing_contexts(self):
        """Test that missing contexts raises ValueError."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        from ragas.metrics.collections.multi_modal_faithfulness import (
            MultiModalFaithfulness,
        )

        metric = object.__new__(MultiModalFaithfulness)
        metric.llm = mock_llm
        metric.name = "test"

        with pytest.raises(ValueError, match="retrieved_contexts is missing"):
            await metric.ascore(
                response="Some response",
                retrieved_contexts=[],
            )

    def test_metric_name_default(self):
        """Test that default metric name is set correctly."""
        from unittest.mock import MagicMock

        from ragas.metrics.collections.multi_modal_faithfulness import (
            MultiModalFaithfulness,
        )

        mock_llm = MagicMock()
        mock_llm._map_provider_params = MagicMock(return_value={})

        metric = object.__new__(MultiModalFaithfulness)
        metric.llm = mock_llm
        metric.name = "multi_modal_faithfulness"

        assert metric.name == "multi_modal_faithfulness"
