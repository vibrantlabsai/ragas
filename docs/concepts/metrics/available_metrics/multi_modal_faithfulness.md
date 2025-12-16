## MultiModalFaithfulness

`MultiModalFaithfulness` metric measures the factual consistency of the generated answer against both visual and textual context. It is calculated from the answer, retrieved textual context, and visual context. The answer is scaled to a (0,1) range, with higher scores indicating better faithfulness.

The generated answer is regarded as faithful if all the claims made in the answer can be inferred from either the visual or textual context provided. To determine this, the response is directly evaluated against the provided contexts, and the faithfulness score is either 0 or 1.

### Example (Recommended - Collections API)

```python
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import MultiModalFaithfulness

# Setup - use a vision-capable model
client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)  # Vision-capable model required

# Create metric instance
metric = MultiModalFaithfulness(llm=llm)

# Evaluate faithfulness
result = await metric.ascore(
    response="The Tesla Model X is an electric SUV.",
    retrieved_contexts=[
        "path/to/tesla_image.jpg",  # Image context
        "Tesla manufactures electric vehicles."  # Text context
    ]
)
print(f"Faithfulness Score: {result.value}")  # 1.0 (faithful) or 0.0 (not faithful)
```

### Example (Legacy API - Deprecated)

!!! warning "Deprecated"
    The legacy API is deprecated and will be removed in a future version. Please migrate to the Collections API shown above.

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import MultiModalFaithfulness

sample = SingleTurnSample(
        user_input="What about the Tesla Model X?",
        response="Cats are cute.",
        retrieved_contexts=[
            "custom_eval/multimodal/images/tesla.jpg"
        ]
    )
scorer = MultiModalFaithfulness()
await scorer.single_turn_ascore(sample)
```

### How It's Calculated

!!! example
    **Question**: What about the Tesla Model X?

    **Context (visual)**:
    - An image of the Tesla Model X (custom_eval/multimodal/images/tesla.jpg)

    **High faithfulness answer**: The Tesla Model X is an electric SUV manufactured by Tesla.

    **Low faithfulness answer**: Cats are cute.

Let's examine how faithfulness was calculated using the low faithfulness answer:

- **Step 1:** Evaluate the generated response against the given contexts.
    - Response: "Cats are cute."

- **Step 2:** Verify if the response can be inferred from the given context.
    - Response: No

- **Step 3:** Use the result to determine the faithfulness score.

    $$
    \text{Faithfulness} = 0
    $$

In this example, the response "Cats are cute" cannot be inferred from the image of the Tesla Model X, so the faithfulness score is 0.

### Supported Context Types

The metric supports multiple types of context inputs:

- **Text contexts**: Plain text strings
- **Image URLs**: HTTP/HTTPS URLs pointing to images
- **Local image paths**: File paths to local images (jpg, png, gif, webp, bmp)
- **Base64 data URIs**: Inline base64-encoded images

### Requirements

- A vision-capable LLM is required (e.g., `gpt-4o`, `gpt-4-vision-preview`, `claude-3-opus`, `gemini-pro-vision`)
- For the Collections API, use `llm_factory` to create the LLM instance
