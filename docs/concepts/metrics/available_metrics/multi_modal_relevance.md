## MultiModalRelevance

`MultiModalRelevance` metric measures the relevance of the generated answer against both visual and textual context. It is calculated from the user input, response, and retrieved contexts (both visual and textual). The answer is scaled to a (0,1) range, with higher scores indicating better relevance.

The generated answer is regarded as relevant if it aligns with the visual or textual context provided. To determine this, the response is directly evaluated against the provided contexts, and the relevance score is either 0 or 1.

### Example (Recommended - Collections API)

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import MultiModalRelevance

# Setup - use a vision-capable model
client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)  # Vision-capable model required

# Create metric instance
metric = MultiModalRelevance(llm=llm)

# Evaluate relevance
result = await metric.ascore(
    user_input="What about the Tesla Model X?",
    response="The Tesla Model X is an electric SUV.",
    retrieved_contexts=[
        "https://example.com/tesla_image.jpg",  # Image context
        "Tesla manufactures electric vehicles.",  # Text context
    ],
)
print(f"Relevance Score: {result.value}")  # 1.0 (relevant) or 0.0 (not relevant)
```

### Example (Legacy API - Deprecated)

!!! warning "Deprecated"
    The legacy API is deprecated and will be removed in a future version. Please migrate to the Collections API shown above.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MultiModalRelevance

sample = SingleTurnSample(
    user_input="What about the Tesla Model X?",
    response="Cats are cute.",
    retrieved_contexts=["https://example.com/tesla_image.jpg"],
)
scorer = MultiModalRelevance()
await scorer.single_turn_ascore(sample)
```

### How It's Calculated

!!! example
    **Question**: What about the Tesla Model X?

    **Context (visual)**:
    - An image of the Tesla Model X (https://example.com/tesla_image.jpg)

    **High relevance answer**: The Tesla Model X is an electric SUV manufactured by Tesla.

    **Low relevance answer**: Cats are cute.

Let's examine how relevance was calculated using the low relevance answer:

- **Step 1:** Evaluate the generated response against the given contexts.
    - Response: "Cats are cute."

- **Step 2:** Verify if the response aligns with the given context.
    - Response: No

- **Step 3:** Use the result to determine the relevance score.

    $$
    \text{Relevance} = 0
    $$

In this example, the response "Cats are cute" does not align with the image of the Tesla Model X, so the relevance score is 0.

### Supported Context Types

The metric supports multiple types of context inputs:

- **Text contexts**: Plain text strings
- **Image URLs**: HTTP/HTTPS URLs pointing to images on the public internet
- **Base64 data URIs**: Inline base64-encoded images
- **Local image paths**: File paths to local images (jpg, png, gif, webp, bmp). Disabled by default, see below

### Image Loading and Security

!!! note "Changed in version 0.4.4"
    Local image paths are no longer loaded by default, and image URLs on internal or private addresses are refused. Earlier versions loaded both. If your contexts use either, the image is now passed to the model as text until you opt in as shown below. This change fixes CVE-2026-6587.

Retrieved contexts usually come from a dataset, so Ragas treats image references as untrusted input:

- **Image URLs** are downloaded only from public internet addresses. A URL that points to `localhost`, a private or link-local address (including cloud metadata endpoints), or that redirects to one, is not fetched.
- **Local image paths** are not read unless you enable local file access and name the single directory that images may be loaded from.

An image that is not loaded is passed to the model as plain text and a message is logged, so a blocked image shows up in the logs rather than as an error.

If your images live on an internal host or on disk, opt in before running the metric:

```python
import ragas.prompt.multi_modal_prompt as image_policy

# Read local images, but only from inside this directory
image_policy.ALLOW_LOCAL_FILE_ACCESS = True
image_policy.ALLOWED_IMAGE_BASE_DIR = "/data/eval_images"

# Allow image URLs on internal or private addresses
image_policy.ALLOW_INTERNAL_TARGETS = True
```

Local paths are resolved relative to `ALLOWED_IMAGE_BASE_DIR`, and a path that resolves outside it is refused.

!!! warning
    Turn these settings on only when you trust every context in the dataset. With them enabled, a context string can make the machine running the evaluation read a local file or call an internal service and send the result to your LLM provider.

### Requirements

- A vision-capable LLM is required (e.g., `gpt-4o`, `gpt-4-vision-preview`, `claude-3-opus`, `gemini-pro-vision`)
- For the Collections API, use `llm_factory` to create the LLM instance
