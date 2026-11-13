from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import typing as t
from io import BytesIO
from urllib.parse import urlparse

import requests
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from PIL import Image
from pydantic import BaseModel
from typing_extensions import TypedDict

from ragas._safe_fetch import BlockedAddressError, fetch_url_bytes
from ragas.callbacks import ChainType, new_group
from ragas.exceptions import RagasOutputParserException
from ragas.prompt.pydantic_prompt import (
    PydanticPrompt,
    RagasOutputParser,
    is_langchain_llm,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

from ragas.llms.base import BaseRagasLLM

# type variables for input and output models
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


# Specific typed dictionaries for message content
class TextContent(TypedDict):
    type: t.Literal["text"]
    text: str


class ImageUrlContent(TypedDict):
    type: t.Literal["image_url"]
    image_url: dict[str, str]


MessageContent = t.Union[TextContent, ImageUrlContent]

logger = logging.getLogger(__name__)

# --- Constants for Security Policy ---
# ALLOW_INTERNAL_TARGETS, ALLOW_LOCAL_FILE_ACCESS and ALLOWED_IMAGE_BASE_DIR are
# read at call time and also govern the collections multimodal metrics
# (ragas.metrics.collections.multi_modal_faithfulness / multi_modal_relevance).

# Allow only HTTP and HTTPS URLs by default
ALLOWED_URL_SCHEMES = {"http", "https"}
# Maximum download size in bytes (e.g., 10MB) - ADJUST AS NEEDED
MAX_DOWNLOAD_SIZE_BYTES = 10 * 1024 * 1024
# Request timeout in seconds - ADJUST AS NEEDED
REQUESTS_TIMEOUT_SECONDS = 10
# Regex to parse data URIs (simplistic, adjust if more complex URIs needed)
DATA_URI_REGEX = re.compile(
    r"^data:(image\/(?:png|jpeg|gif|webp));base64,([a-zA-Z0-9+/=]+)$"
)

COMMON_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# --- OPTIONAL: Local File Access Configuration ---
# Set to True ONLY if local file access is absolutely required and understood.
ALLOW_LOCAL_FILE_ACCESS = False  # <<< SECURITY: Default to False

ALLOW_INTERNAL_TARGETS = False  # <<< SECURITY: Default to False


# Define the *absolute* path to the ONLY directory from which local images can be loaded.
# Ensure this directory is not web-accessible and contains only safe images.
# Example: ALLOWED_IMAGE_BASE_DIR = "/var/app/allowed_images"
ALLOWED_IMAGE_BASE_DIR = (
    None  # <<< SECURITY: Must be configured if ALLOW_LOCAL_FILE_ACCESS=True
)
# Maximum local file size - ADJUST AS NEEDED
MAX_LOCAL_FILE_SIZE_BYTES = 10 * 1024 * 1024


class ImageTextPrompt(PydanticPrompt, t.Generic[InputModel, OutputModel]):
    def _generate_examples(self):
        if self.examples:
            example_strings = []
            for e in self.examples:
                input_data, output_data = e
                example_strings.append(
                    self.instruction
                    + "\n"
                    + "input: "
                    + input_data.model_dump_json(indent=4)
                    + "\n"
                    + "output: "
                    + output_data.model_dump_json(indent=4)
                )

            return (
                "Some examples are provided below with only text context, but please do use any images for context if they are provided.\n"
                + "\n\n".join(example_strings)
            )
        # if no examples are provided
        else:
            return ""

    def to_prompt_value(self, data: t.Optional[InputModel] = None):
        text = [
            self._generate_instruction(),
            self._generate_output_signature(),
            self._generate_examples(),
            "Now perform the above instruction with the following",
        ] + data.to_string_list()  # type: ignore
        return ImageTextPromptValue(items=text)

    async def generate_multiple(
        self,
        llm: t.Union[BaseRagasLLM, BaseLanguageModel],
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
        retries_left: int = 3,
    ) -> t.List[OutputModel]:
        """
        Generate multiple outputs using the provided language model and input data.

        Parameters
        ----------
        llm : BaseRagasLLM
            The language model to use for generation.
        data : InputModel
            The input data for generation.
        n : int, optional
            The number of outputs to generate. Default is 1.
        temperature : float, optional
            The temperature parameter for controlling randomness in generation.
        stop : List[str], optional
            A list of stop sequences to end generation.
        callbacks : Callbacks, optional
            Callback functions to be called during the generation process.

        Returns
        -------
        List[OutputModel]
            A list of generated outputs.

        Raises
        ------
        RagasOutputParserException
            If there's an error parsing the output.
        """
        callbacks = callbacks or []
        processed_data = self.process_input(data)
        prompt_rm, prompt_cb = new_group(
            name=self.name,
            inputs={"data": processed_data},
            callbacks=callbacks,
            metadata={"type": ChainType.RAGAS_PROMPT},
        )
        prompt_value = self.to_prompt_value(processed_data)

        # Handle both LangChain LLMs and Ragas LLMs
        # LangChain LLMs have agenerate() for async, generate() for sync
        # Ragas LLMs have generate() as async method
        if is_langchain_llm(llm):
            # This is a LangChain LLM - use agenerate_prompt()
            langchain_llm = t.cast(BaseLanguageModel, llm)
            resp = await langchain_llm.agenerate_prompt(
                [prompt_value],
                stop=stop,
                callbacks=prompt_cb,
            )
        else:
            # This is a Ragas LLM - use generate()
            ragas_llm = t.cast(BaseRagasLLM, llm)
            resp = await ragas_llm.generate(
                prompt_value,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=prompt_cb,
            )

        output_models = []
        parser = RagasOutputParser(pydantic_object=self.output_model)  # type: ignore
        for i in range(n):
            output_string = resp.generations[0][i].text
            try:
                # For the parser, we need a BaseRagasLLM, so if it's a LangChain LLM, we need to handle this
                if is_langchain_llm(llm):
                    # Skip parsing retry for LangChain LLMs since parser expects BaseRagasLLM
                    answer = self.output_model.model_validate_json(output_string)
                else:
                    ragas_llm = t.cast(BaseRagasLLM, llm)
                    answer = await parser.parse_output_string(
                        output_string=output_string,
                        prompt_value=prompt_value,  # type: ignore
                        llm=ragas_llm,
                        callbacks=prompt_cb,
                        retries_left=retries_left,
                    )
                processed_output = self.process_output(answer, data)  # type: ignore
                output_models.append(processed_output)
            except RagasOutputParserException as e:
                prompt_rm.on_chain_error(error=e)
                logger.error("Prompt %s failed to parse output: %s", self.name, e)
                raise e

        prompt_rm.on_chain_end({"output": output_models})
        return output_models


class ImageTextPromptValue(PromptValue):
    items: t.List[str]

    def __len__(self):
        """Return the number of items."""
        return len(self.items)

    def to_messages(self) -> t.List[BaseMessage]:
        """
        Converts items into a list of BaseMessages, securely processing potential
        image references (Base64 data URIs or allowed URLs).
        """
        messages_content = []
        for item in self.items:
            processed_item = self._securely_process_item(item)
            messages_content.append(processed_item)

        # Filter out potential None values if _securely_process_item indicates failure
        valid_messages_content = [m for m in messages_content if m is not None]

        # Only create HumanMessage if there's valid content
        if valid_messages_content:
            return [HumanMessage(content=valid_messages_content)]
        else:
            # Return empty list or handle as appropriate if all items failed processing
            return []

    def _securely_process_item(self, item: str) -> t.Optional[MessageContent]:
        """
        Securely determines if an item is text, a valid image data URI,
        or a fetchable image URL according to policy. Returns the appropriate
        message dictionary structure or None if invalid/unsafe.
        """
        if not isinstance(item, str):
            logger.warning(f"Processing non-string item as text: {type(item)}")
            return self._get_text_payload(str(item))

        # 1. Check for Base64 Data URI
        image_data = self._try_process_base64_uri(item)
        if image_data:
            return self._get_image_payload(
                image_data["mime_type"], image_data["encoded_data"]
            )

        # 2. Check for Allowed URL
        image_data = self._try_process_allowed_url(item)
        if image_data:
            return self._get_image_payload(
                image_data["mime_type"], image_data["encoded_data"]
            )

        # 3. Check for Allowed Local File Path (Optional & Discouraged)
        # <<< MODIFICATION START >>>
        # Only attempt local file processing if the feature is enabled AND
        # the item heuristically looks like an image path based on its extension.
        if ALLOW_LOCAL_FILE_ACCESS and self._looks_like_image_path(item):
            # <<< MODIFICATION END >>>
            image_data = self._try_process_local_file(item)
            if image_data:
                # Ensure we use the mime_type verified from content, not from heuristic
                return self._get_image_payload(
                    image_data["mime_type"], image_data["encoded_data"]
                )

        # 4. If none of the above, treat as text
        return self._get_text_payload(item)

    def _looks_like_image_path(self, item: str) -> bool:
        """
        A simple heuristic to check if a string looks like a potential image file path
        based on its extension. This is NOT for security validation, only to avoid
        unnecessary filesystem checks on instruction text when local file access is enabled.
        """
        if not isinstance(item, str) or not item:
            return False
        # Check if the string ends with one of the common image extensions (case-insensitive)
        # Ignores potential query/fragment parts for this basic check
        path_part = urlparse(item).path
        _, ext = os.path.splitext(path_part)
        return ext.lower() in COMMON_IMAGE_EXTENSIONS

    def _get_text_payload(self, text: str) -> TextContent:
        """Returns the standard payload for text content."""
        return {"type": "text", "text": text}

    def _get_image_payload(self, mime_type: str, encoded_image: str) -> ImageUrlContent:
        """Returns the standard payload for image content."""
        # Ensure mime_type is safe and starts with "image/"
        if not mime_type or not mime_type.lower().startswith("image/"):
            # Fallback or default if mime_type validation failed earlier
            safe_mime_type = "image/jpeg"  # Or consider raising an error
            logger.warning(
                f"Invalid or missing mime_type '{mime_type}', defaulting to {safe_mime_type}"
            )
        else:
            safe_mime_type = mime_type.lower()  # Use validated mime type

        return {
            "type": "image_url",
            "image_url": {"url": f"data:{safe_mime_type};base64,{encoded_image}"},
        }

    def _try_process_base64_uri(self, item: str) -> t.Optional[dict]:
        """
        Checks if the item is a valid data:image/...;base64 URI.
        Returns dict with 'mime_type' and 'encoded_data' or None.
        """
        match = DATA_URI_REGEX.match(item)
        if match:
            mime_type = match.group(1)
            encoded_data = match.group(2)
            # Optional: Add deeper validation by trying to decode and check magic bytes
            try:
                # Try decoding to validate base64 format
                base64.b64decode(encoded_data)
                # Optional: Use Pillow to verify it's a valid image format
                # try:
                #     img = Image.open(BytesIO(decoded_bytes))
                #     img.verify() # Check for corruption
                #     # could check img.format matches mime_type roughly
                # except Exception:
                #      logger.warning(f"Base64 data for {mime_type} is not a valid image.")
                #      return None
                return {"mime_type": mime_type, "encoded_data": encoded_data}
            except (binascii.Error, ValueError) as e:
                logger.warning(f"Failed to decode base64 string: {e}")
                return None
        return None

    def _try_process_allowed_url(self, item: str) -> t.Optional[dict]:
        """
        Checks if the item is a URL with an allowed scheme (http/https).
        If so, attempts to download, validate, and encode the image.
        Returns dict with 'mime_type' and 'encoded_data' or None.
        """
        try:
            parsed_url = urlparse(item)
            if parsed_url.scheme in ALLOWED_URL_SCHEMES:
                # URL seems plausible, attempt download and validation
                return self._download_validate_and_encode(item)
        except ValueError:
            # Invalid URL format
            pass
        return None

    def _download_validate_and_encode(self, url: str) -> t.Optional[dict]:
        """
        Downloads content from URL, validates target IP, size and type, encodes if valid image.
        Uses 'requests' library for better control.
        """
        try:
            # 1-3. Download with the SSRF guard: the target of the request and of
            # every redirect hop is validated, and the size limit is enforced.
            image_data = BytesIO(
                fetch_url_bytes(
                    url,
                    timeout=REQUESTS_TIMEOUT_SECONDS,
                    max_bytes=MAX_DOWNLOAD_SIZE_BYTES,
                    allow_internal=ALLOW_INTERNAL_TARGETS,
                )
            )

            # 4. Validate content using Pillow
            try:
                with Image.open(image_data) as img:
                    img.verify()  # Checks if image data is corrupt
                    # Reload image after verify()
                    image_data.seek(0)
                    with Image.open(image_data) as img_reloaded:
                        img_format = (
                            img_reloaded.format
                        )  # Get actual format (JPEG, PNG, etc.)
                        if not img_format:
                            logger.error(
                                f"Could not determine image format for URL {url}."
                            )
                            return None
                        verified_mime_type = f"image/{img_format.lower()}"

                # 5. Encode validated image data
                image_data.seek(0)
                encoded_string = base64.b64encode(image_data.read()).decode("utf-8")
                return {"mime_type": verified_mime_type, "encoded_data": encoded_string}

            except (Image.UnidentifiedImageError, SyntaxError, IOError) as img_err:
                logger.error(
                    f"Content validation failed for URL {url}. Not a valid image. Error: {img_err}"
                )
                return None

        except BlockedAddressError as ssrf_err:
            logger.error(f"SSRF check: blocking request to URL {url}: {ssrf_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Failed to download image from URL {url}: {req_err}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred processing URL {url}: {e}")
            return None

    def _try_process_local_file(self, item: str) -> t.Optional[dict]:
        """
        (Optional) Checks if item is an allowed local file path.
        Reads, validates, and encodes the image if valid.
        Returns dict with 'mime_type' and 'encoded_data' or None.
        THIS IS HIGHLY DISCOURAGED due to security risks.
        """
        if not ALLOW_LOCAL_FILE_ACCESS:
            return None  # Explicitly disabled

        if not ALLOWED_IMAGE_BASE_DIR or not os.path.isdir(ALLOWED_IMAGE_BASE_DIR):
            logger.critical(
                "Local file access enabled, but ALLOWED_IMAGE_BASE_DIR is not configured or invalid."
            )
            return None

        try:
            # Basic check: prevent absolute paths or obvious traversals if base dir is relative (though base should be absolute)
            if os.path.isabs(item) or ".." in item.split(os.path.sep):
                logger.warning(
                    f"Local path '{item}' appears absolute or contains traversal."
                )
                return None

            # Construct the full path relative to the allowed base directory
            candidate_path = os.path.join(ALLOWED_IMAGE_BASE_DIR, item)

            # CRITICAL: Normalize the path and verify it's still within the allowed directory
            # This prevents various traversal bypasses.
            abs_candidate_path = os.path.abspath(candidate_path)
            abs_allowed_dir = os.path.abspath(ALLOWED_IMAGE_BASE_DIR)

            if (
                os.path.commonprefix([abs_candidate_path, abs_allowed_dir])
                != abs_allowed_dir
            ):
                logger.error(
                    f"Path traversal detected: '{item}' resolves outside allowed directory '{ALLOWED_IMAGE_BASE_DIR}'."
                )
                return None

            # Check if the path exists and is a file
            if not os.path.isfile(abs_candidate_path):
                logger.warning(
                    f"Local file path '{abs_candidate_path}' does not exist or is not a file."
                )
                return None

            # Check file size limit BEFORE reading
            file_size = os.path.getsize(abs_candidate_path)
            if file_size > MAX_LOCAL_FILE_SIZE_BYTES:
                logger.error(
                    f"Local file '{abs_candidate_path}' size {file_size} exceeds limit {MAX_LOCAL_FILE_SIZE_BYTES}."
                )
                return None

            # Read and validate the file content
            with open(abs_candidate_path, "rb") as f:
                file_content = f.read()

            # Validate content using Pillow
            try:
                with Image.open(BytesIO(file_content)) as img:
                    img.verify()
                    # Reload after verify
                    with Image.open(BytesIO(file_content)) as img_reloaded:
                        img_format = img_reloaded.format
                        if not img_format:
                            logger.error(
                                f"Could not determine image format for file {abs_candidate_path}."
                            )
                            return None
                        verified_mime_type = f"image/{img_format.lower()}"

                # Encode validated image data
                encoded_string = base64.b64encode(file_content).decode("utf-8")
                return {"mime_type": verified_mime_type, "encoded_data": encoded_string}

            except (Image.UnidentifiedImageError, SyntaxError, IOError) as img_err:
                logger.error(
                    f"Content validation failed for file {abs_candidate_path}. Not a valid image. Error: {img_err}"
                )
                return None

        except Exception as e:
            logger.error(
                f"An unexpected error occurred processing local file path '{item}': {e}"
            )
            return None

    def to_string(self):
        # This needs adjustment if it relies on the old `is_image`
        # A safer version might just concatenate text or use a placeholder
        # For now, let's assume it can just join the original items for a basic representation
        return " ".join(str(item) for item in self.items).strip()
