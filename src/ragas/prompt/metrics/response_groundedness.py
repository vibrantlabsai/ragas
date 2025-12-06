"""Response groundedness prompts - V1-identical converted to functions."""


def response_groundedness_judge1_prompt(response: str, context: str) -> str:
    """
    V1-identical response groundedness judge 1 prompt - matches template_groundedness1 exactly.

    Args:
        response: The response/assertion to evaluate for groundedness
        context: The context to evaluate the response against

    Returns:
        V1-identical prompt string for the LLM
    """
    return f"""### Instruction

You are a world class expert designed to evaluate the groundedness of an assertion.
You will be provided with an assertion and a context.
Your task is to determine if the assertion is supported by the context.
Follow the instructions below:
A. If there is no context or no assertion or context is empty or assertion is empty, say 0.
B. If the assertion is not supported by the context, say 0.
C. If the assertion is partially supported by the context, say 1.
D. If the assertion is fully supported by the context, say 2.
You must provide a rating of 0, 1, or 2, nothing else.

### Context:
<{context}>

### Assertion:
<{response}>

Analyzing Context and Response, the Groundedness score is """


def response_groundedness_judge2_prompt(response: str, context: str) -> str:
    """
    V1-identical response groundedness judge 2 prompt - matches template_groundedness2 exactly.

    Args:
        response: The response/assertion to evaluate for groundedness
        context: The context to evaluate the response against

    Returns:
        V1-identical prompt string for the LLM
    """
    return f"""As a specialist in assessing the strength of connections between statements and their given contexts, I will evaluate the level of support an assertion receives from the provided context. Follow these guidelines:

* If the assertion is not supported or context is empty or assertion is empty, assign a score of 0.
* If the assertion is partially supported, assign a score of 1.
* If the assertion is fully supported, assign a score of 2.

I will provide a rating of 0, 1, or 2, without any additional information.

---
**Context:**
[{context}]

**Assertion:**
[{response}]

Do not explain. Based on the provided context and response, the Groundedness score is:"""
