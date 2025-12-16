"""Utility functions and prompt classes for MultiModalFaithfulness metric."""

import base64
import binascii
import logging
import os
import re
import typing as t
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants for security/processing
ALLOWED_URL_SCHEMES = {"http", "https"}
MAX_DOWNLOAD_SIZE_BYTES = 10 * 1024 * 1024
REQUESTS_TIMEOUT_SECONDS = 10
DATA_URI_REGEX = re.compile(
    r"^data:(image\/(?:png|jpeg|gif|webp));base64,([a-zA-Z0-9+/=]+)$"
)
COMMON_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


class MultiModalFaithfulnessInput(BaseModel):
    """Input model for multimodal faithfulness evaluation."""

    response: str = Field(..., description="The response to evaluate for faithfulness")
    retrieved_contexts: t.List[str] = Field(
        ...,
        description="List of retrieved contexts (text or image paths/URLs)",
    )


class MultiModalFaithfulnessOutput(BaseModel):
    """Output model for multimodal faithfulness evaluation."""

    faithful: bool = Field(
        ...,
        description="True if the response is faithful to the contexts, False otherwise",
    )
    reason: str = Field(
        default="",
        description="Explanation for the faithfulness verdict",
    )


# Image processing utilities (adapted from multi_modal_prompt.py)


def is_image_path_or_url(item: str) -> bool:
    """Check if a string looks like an image path or URL."""
    if not isinstance(item, str) or not item:
        return False

    # Check for base64 data URI
    if DATA_URI_REGEX.match(item):
        return True

    # Check for URL
    try:
        parsed = urlparse(item)
        if parsed.scheme in ALLOWED_URL_SCHEMES:
            path_part = parsed.path
            _, ext = os.path.splitext(path_part)
            if ext.lower() in COMMON_IMAGE_EXTENSIONS:
                return True
            # Could be an image URL without extension
            return True if parsed.scheme in ALLOWED_URL_SCHEMES else False
    except ValueError:
        pass

    # Check for local file path with image extension
    _, ext = os.path.splitext(item)
    if ext.lower() in COMMON_IMAGE_EXTENSIONS:
        return True

    return False


def process_image_to_base64(item: str) -> t.Optional[t.Dict[str, str]]:
    """
    Process an image reference (URL, base64, or path) to base64 data.

    Returns dict with 'mime_type' and 'encoded_data' or None if not an image.
    """
    # Try base64 data URI first
    result = _try_process_base64_uri(item)
    if result:
        return result

    # Try URL
    result = _try_process_url(item)
    if result:
        return result

    # Try local file
    result = _try_process_local_file(item)
    if result:
        return result

    return None


def _try_process_base64_uri(item: str) -> t.Optional[t.Dict[str, str]]:
    """Check if item is a base64 data URI and extract the data."""
    match = DATA_URI_REGEX.match(item)
    if match:
        mime_type = match.group(1)
        encoded_data = match.group(2)
        try:
            base64.b64decode(encoded_data)
            return {"mime_type": mime_type, "encoded_data": encoded_data}
        except (binascii.Error, ValueError) as e:
            logger.warning(f"Failed to decode base64 string: {e}")
            return None
    return None


def _try_process_url(item: str) -> t.Optional[t.Dict[str, str]]:
    """Download and process image from URL."""
    try:
        parsed_url = urlparse(item)
        if parsed_url.scheme not in ALLOWED_URL_SCHEMES:
            return None

        response = requests.get(
            item,
            timeout=REQUESTS_TIMEOUT_SECONDS,
            stream=True,
        )
        response.raise_for_status()

        # Check content length
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
            logger.error(f"URL {item} content too large")
            return None

        # Download and validate
        image_data = BytesIO()
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > MAX_DOWNLOAD_SIZE_BYTES:
                logger.error(f"URL {item} download exceeded size limit")
                return None
            image_data.write(chunk)

        image_data.seek(0)

        # Validate with PIL
        try:
            with Image.open(image_data) as img:
                img.verify()
                image_data.seek(0)
                with Image.open(image_data) as img_reloaded:
                    img_format = img_reloaded.format
                    if not img_format:
                        return None
                    verified_mime_type = f"image/{img_format.lower()}"

            image_data.seek(0)
            encoded_string = base64.b64encode(image_data.read()).decode("utf-8")
            return {"mime_type": verified_mime_type, "encoded_data": encoded_string}
        except (Image.UnidentifiedImageError, SyntaxError, IOError):
            return None

    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


def _try_process_local_file(item: str) -> t.Optional[t.Dict[str, str]]:
    """Process local image file."""
    try:
        # Check if file exists
        if not os.path.isfile(item):
            return None

        # Check file size
        file_size = os.path.getsize(item)
        if file_size > MAX_DOWNLOAD_SIZE_BYTES:
            logger.error(f"Local file {item} too large")
            return None

        # Read and validate
        with open(item, "rb") as f:
            file_content = f.read()

        try:
            with Image.open(BytesIO(file_content)) as img:
                img.verify()
                with Image.open(BytesIO(file_content)) as img_reloaded:
                    img_format = img_reloaded.format
                    if not img_format:
                        return None
                    verified_mime_type = f"image/{img_format.lower()}"

            encoded_string = base64.b64encode(file_content).decode("utf-8")
            return {"mime_type": verified_mime_type, "encoded_data": encoded_string}
        except (Image.UnidentifiedImageError, SyntaxError, IOError):
            return None

    except Exception:
        return None


def build_multimodal_message_content(
    instruction: str,
    response: str,
    retrieved_contexts: t.List[str],
) -> t.List[t.Dict[str, t.Any]]:
    """
    Build multimodal message content for the LLM.

    Args:
        instruction: The evaluation instruction
        response: The response to evaluate
        retrieved_contexts: List of contexts (text or image references)

    Returns:
        List of content blocks for the message
    """
    content: t.List[t.Dict[str, t.Any]] = []

    # Add instruction and response
    prompt_text = f"""{instruction}

Response to evaluate:
{response}

Retrieved contexts:
"""
    content.append({"type": "text", "text": prompt_text})

    # Process each context
    for i, ctx in enumerate(retrieved_contexts):
        # Try to process as image
        image_data = process_image_to_base64(ctx)

        if image_data:
            # Add as image
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['encoded_data']}"
                    },
                }
            )
            content.append({"type": "text", "text": f"[Image context {i + 1}]"})
        else:
            # Add as text
            content.append({"type": "text", "text": f"Context {i + 1}: {ctx}"})

    # Add closing instruction
    content.append(
        {
            "type": "text",
            "text": "\n\nBased on the above contexts (both visual and textual), determine if the response is faithful. A response is faithful if all claims can be inferred from the provided contexts.",
        }
    )

    return content


# Instruction for the prompt
MULTIMODAL_FAITHFULNESS_INSTRUCTION = """You are evaluating whether a response is faithful to the provided context information.

A response is considered FAITHFUL if:
- All claims in the response can be directly inferred from the visual or textual context
- The response does not contain information that contradicts the context
- The response does not hallucinate facts not present in the context

A response is considered NOT FAITHFUL if:
- It contains claims that cannot be verified from the context
- It contradicts information in the context
- It makes up facts not supported by the context

You must evaluate faithfulness based on BOTH visual (images) and textual context if provided."""
