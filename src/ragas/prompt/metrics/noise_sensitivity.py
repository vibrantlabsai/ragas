"""Noise Sensitivity prompts - V1-identical using exact PydanticPrompt.to_string() output."""

import json
import typing as t


def nli_statement_prompt(context: str, statements: t.List[str]) -> str:
    """
    V1-identical NLI statement evaluation - matches PydanticPrompt.to_string() exactly.

    Args:
        context: The context to evaluate statements against
        statements: The statements to judge for faithfulness

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_context = json.dumps(context)
    safe_statements = json.dumps(statements, indent=4).replace("\n", "\n    ")

    return f"""Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"$defs": {{"StatementFaithfulnessAnswer": {{"properties": {{"statement": {{"description": "the original statement, word-by-word", "title": "Statement", "type": "string"}}, "reason": {{"description": "the reason of the verdict", "title": "Reason", "type": "string"}}, "verdict": {{"description": "the verdict(0/1) of the faithfulness.", "title": "Verdict", "type": "integer"}}}}, "required": ["statement", "reason", "verdict"], "title": "StatementFaithfulnessAnswer", "type": "object"}}}}, "properties": {{"statements": {{"items": {{"$ref": "#/$defs/StatementFaithfulnessAnswer"}}, "title": "Statements", "type": "array"}}}}, "required": ["statements"], "title": "NLIStatementOutput", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
    "statements": [
        "John is majoring in Biology.",
        "John is taking a course on Artificial Intelligence.",
        "John is a dedicated student.",
        "John has a part-time job."
    ]
}}
Output: {{
    "statements": [
        {{
            "statement": "John is majoring in Biology.",
            "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
            "verdict": 0
        }},
        {{
            "statement": "John is taking a course on Artificial Intelligence.",
            "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
            "verdict": 0
        }},
        {{
            "statement": "John is a dedicated student.",
            "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
            "verdict": 1
        }},
        {{
            "statement": "John has a part-time job.",
            "reason": "There is no information given in the context about John having a part-time job.",
            "verdict": 0
        }}
    ]
}}

Example 2
Input: {{
    "context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
    "statements": [
        "Albert Einstein was a genius."
    ]
}}
Output: {{
    "statements": [
        {{
            "statement": "Albert Einstein was a genius.",
            "reason": "The context and statement are unrelated",
            "verdict": 0
        }}
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "context": {safe_context},
    "statements": {safe_statements}
}}
Output: """
