"""Common prompts shared across multiple metrics."""

import json
import typing as t


def statement_generator_prompt(question: str, answer: str) -> str:
    """
    V1-identical statement generator - matches PydanticPrompt.to_string() exactly.

    Args:
        question: The question being answered
        answer: The answer text to break down into statements

    Returns:
        V1-identical prompt string for the LLM
    """
    # Format inputs exactly like V1's model_dump_json(indent=4, exclude_none=True)
    safe_question = json.dumps(question)
    safe_answer = json.dumps(answer)

    return f"""Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{{"properties": {{"statements": {{"description": "The generated statements", "items": {{"type": "string"}}, "title": "Statements", "type": "array"}}}}, "required": ["statements"], "title": "StatementGeneratorOutput", "type": "object"}}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

--------EXAMPLES-----------
Example 1
Input: {{
    "question": "Who was Albert Einstein and what is he best known for?",
    "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."
}}
Output: {{
    "statements": [
        "Albert Einstein was a German-born theoretical physicist.",
        "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
        "Albert Einstein was best known for developing the theory of relativity.",
        "Albert Einstein made important contributions to the development of the theory of quantum mechanics."
    ]
}}
-----------------------------

Now perform the same with the following input
input: {{
    "question": {safe_question},
    "answer": {safe_answer}
}}
Output: """


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
            "reason": "John's major is explicitly stated as Computer Science, not Biology.",
            "verdict": 0
        }},
        {{
            "statement": "John is taking a course on Artificial Intelligence.",
            "reason": "The context mentions courses in Data Structures, Algorithms, and Database Management, but does not mention Artificial Intelligence.",
            "verdict": 0
        }},
        {{
            "statement": "John is a dedicated student.",
            "reason": "The context states that John is a diligent student who spends a significant amount of time studying and completing assignments.",
            "verdict": 1
        }},
        {{
            "statement": "John has a part-time job.",
            "reason": "There is no information in the context about John having a part-time job.",
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
