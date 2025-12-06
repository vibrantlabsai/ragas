"""Factual correctness prompts - V1-identical converted to functions."""

import json


def claim_decomposition_prompt(
    response: str, atomicity: str = "low", coverage: str = "low"
) -> str:
    """
    V1-identical claim decomposition prompt with configurable atomicity/coverage.

    Args:
        response: The response text to break down into claims
        atomicity: Level of atomicity ("low" or "high")
        coverage: Level of coverage ("low" or "high")

    Returns:
        V1-identical prompt string for the LLM
    """
    safe_response = json.dumps(response)

    # Select examples based on atomicity and coverage configuration
    if atomicity == "low" and coverage == "low":
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": ["Charles Babbage was a mathematician and philosopher."]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German physicist.",
                        "Albert Einstein developed relativity and contributed to quantum mechanics.",
                    ]
                },
            },
        ]
    elif atomicity == "low" and coverage == "high":
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": [
                        "Charles Babbage was a French mathematician, philosopher, and food critic."
                    ]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    ]
                },
            },
        ]
    elif atomicity == "high" and coverage == "low":
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": [
                        "Charles Babbage was a mathematician.",
                        "Charles Babbage was a philosopher.",
                    ]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity.",
                    ]
                },
            },
        ]
    else:  # high atomicity, high coverage
        examples = [
            {
                "input": {
                    "response": "Charles Babbage was a French mathematician, philosopher, and food critic."
                },
                "output": {
                    "claims": [
                        "Charles Babbage was a mathematician.",
                        "Charles Babbage was a philosopher.",
                        "Charles Babbage was a food critic.",
                        "Charles Babbage was French.",
                    ]
                },
            },
            {
                "input": {
                    "response": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics."
                },
                "output": {
                    "claims": [
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity.",
                        "Albert Einstein contributed to the development of quantum mechanics.",
                    ]
                },
            },
        ]

    # Build examples string
    examples_str = "\n".join(
        [
            f"""Example {i + 1}
Input: {json.dumps(ex["input"], indent=4)}
Output: {json.dumps(ex["output"], indent=4)}"""
            for i, ex in enumerate(examples)
        ]
    )

    return f"""Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
Follow the level of atomicity and coverage as shown in the examples.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"claims": {{"description": "Decomposed Claims", "items": {{"type": "string"}}, "title": "Claims", "type": "array"}}}}, "required": ["claims"], "title": "ClaimDecompositionOutput", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
{examples_str}
-----------------------------

Now perform the same with the following input
input: {{
    "response": {safe_response}
}}
Output: """
