from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import os
import re
import socket
import typing as t
from io import BytesIO
from urllib.parse import urlparse

import requests
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from PIL import Image
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict

from ragas.callbacks import ChainType, new_group
from ragas.exceptions import RagasOutputParserException
from ragas.prompt.pydantic_prompt import (
    PydanticPrompt,
    RagasOutputParser,
    is_langchain_llm,
)
from ragas.prompt.utils import extract_json

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

DISALLOWED_IP_CHECKS = {"is_loopback", "is_private", "is_link_local", "is_reserved"}


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
                    try:
                        answer = self.output_model.model_validate_json(output_string)
                    except ValidationError:
                        jsonstr = extract_json(output_string)
                        answer = self.output_model.model_validate_json(jsonstr)
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
            # <<< SSRF CHECK START >>>
            parsed_url = urlparse(url)
            if not parsed_url.hostname:
                logger.error(
                    f"Could not extract hostname from URL '{url}' for SSRF check."
                )
                return None

            if not self._is_safe_url_target(parsed_url.hostname):
                # Logging is handled within _is_safe_url_target
                return None
            # <<< SSRF CHECK END >>>

            # Proceed with the request only if the target IP check passed
            response = requests.get(
                url,
                timeout=REQUESTS_TIMEOUT_SECONDS,
                stream=True,
                # IMPORTANT CAVEAT: Redirects can bypass this initial check.
                # An initial safe URL could redirect to an internal one.
                # Setting allow_redirects=False is safer but may break legitimate uses.
                # Handling redirects manually with re-checks is complex.
                # Consider the risk profile. Defaulting to allow_redirects=True for now.
                allow_redirects=True,
            )
            response.raise_for_status()  # Check for HTTP errors (4xx, 5xx)

            # 1. Check Content-Type header (as a hint, not definitive)
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                logger.warning(f"URL {url} Content-Type '{content_type}' is not image.")
                # Allow processing to continue, but rely on content validation later
                # return None # uncomment if strict header check desired

            # 2. Check Content-Length header (if available) against limit
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
                logger.error(
                    f"URL {url} content length {content_length} exceeds limit {MAX_DOWNLOAD_SIZE_BYTES}."
                )
                return None

            # 3. Download content incrementally, enforcing size limit
            image_data = BytesIO()
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > MAX_DOWNLOAD_SIZE_BYTES:
                    logger.error(
                        f"URL {url} download size exceeded limit {MAX_DOWNLOAD_SIZE_BYTES} during streaming."
                    )
                    return None
                image_data.write(chunk)

            image_data.seek(0)  # Rewind buffer for reading

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

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Failed to download image from URL {url}: {req_err}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred processing URL {url}: {e}")
            return None

    def _is_safe_url_target(self, url_hostname: str) -> bool:
        """
        Resolves the URL hostname to IP addresses and checks if any fall into
        disallowed categories (loopback, private, reserved, link-local)
        to prevent SSRF attacks against internal networks.

        Args:
            url_hostname: The hostname extracted from the URL.

        Returns:
            True if all resolved IPs are considered safe (e.g., public),
            False if any resolved IP is disallowed or resolution fails.
        """
        if ALLOW_INTERNAL_TARGETS:
            # Bypass check if explicitly allowed (dangerous!)
            logger.warning(
                "SSRF IP address check bypassed due to ALLOW_INTERNAL_TARGETS=True"
            )
            return True

        try:
            # Use getaddrinfo for robust resolution (handles IPv4/IPv6)
            # The flags ensure we get canonical names and prevent certain resolution loops if needed,
            # though default flags are often sufficient. Using AF_UNSPEC gets both IPv4 and IPv6 if available.
            addrinfo_results = socket.getaddrinfo(
                url_hostname, None, family=socket.AF_UNSPEC
            )
            # Example result: [(<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_STREAM: 1>, 6, '', ('93.184.216.34', 0))]

            if not addrinfo_results:
                logger.error(
                    f"SSRF check: DNS resolution failed for hostname '{url_hostname}' (no results)"
                )
                return False

            for family, type, proto, canonname, sockaddr in addrinfo_results:
                ip_address_str = sockaddr[
                    0
                ]  # IP address is the first element of the sockaddr tuple
                try:
                    ip = ipaddress.ip_address(ip_address_str)

                    # Check against disallowed types using the policy
                    for check_name in DISALLOWED_IP_CHECKS:
                        # Dynamically call the check method (e.g., ip.is_loopback)
                        is_disallowed_type = getattr(ip, check_name, False)
                        if is_disallowed_type:
                            logger.error(
                                f"SSRF check: Hostname '{url_hostname}' resolved to disallowed IP '{ip_address_str}' ({check_name}=True). Blocking request."
                            )
                            return False

                    # Optional: Log allowed IPs for debugging if needed
                    # logger.debug(f"SSRF check: Hostname '{url_hostname}' resolved to allowed IP '{ip_address_str}'")

                except ValueError as ip_err:
                    logger.error(
                        f"SSRF check: Error parsing resolved IP address '{ip_address_str}' for hostname '{url_hostname}': {ip_err}"
                    )
                    # Treat parsing errors as unsafe
                    return False

            # If we looped through all resolved IPs and none were disallowed
            return True

        except socket.gaierror as dns_err:
            logger.error(
                f"SSRF check: DNS resolution error for hostname '{url_hostname}': {dns_err}"
            )
            return False
        except Exception as e:
            # Catch unexpected errors during resolution/checking
            logger.error(
                f"SSRF check: Unexpected error checking hostname '{url_hostname}': {e}"
            )
            return False

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
