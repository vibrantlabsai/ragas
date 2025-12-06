"""Answer Accuracy prompts - Convert NVIDIA dual-judge templates to function format."""

import json


def answer_accuracy_judge1_prompt(
    query: str, user_answer: str, reference_answer: str
) -> str:
    """
    First judge template for answer accuracy evaluation.

    Uses JSON structured output for reliable parsing.

    Args:
        query: The original question
        user_answer: The response to evaluate
        reference_answer: The ground truth reference

    Returns:
        Prompt string for structured JSON rating (0, 2, or 4)
    """
    safe_query = json.dumps(query)
    safe_user_answer = json.dumps(user_answer)
    safe_reference_answer = json.dumps(reference_answer)

    return f"""Instruction: You are a world class state of the art assistant for rating a User Answer given a Question. The Question is completely answered by the Reference Answer.
Say 4, if User Answer is full contained and equivalent to Reference Answer in all terms, topics, numbers, metrics, dates and units.
Say 2, if User Answer is partially contained and almost equivalent to Reference Answer in all terms, topics, numbers, metrics, dates and units.
Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics, numbers, metrics, dates and units or the User Answer do not answer the question.
Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.
Return your response as JSON in this format: {{"rating": X}} where X is 0, 2, or 4.

### Question: {safe_query}
### User Answer: {safe_user_answer}
### Reference Answer: {safe_reference_answer}
The rating is:"""


def answer_accuracy_judge2_prompt(
    query: str, user_answer: str, reference_answer: str
) -> str:
    """
    Second judge template for answer accuracy evaluation.

    Uses JSON structured output for reliable parsing.

    Args:
        query: The original question
        user_answer: The response to evaluate
        reference_answer: The ground truth reference

    Returns:
        Prompt string for structured JSON rating (0, 2, or 4)
    """
    safe_query = json.dumps(query)
    safe_user_answer = json.dumps(user_answer)
    safe_reference_answer = json.dumps(reference_answer)

    return f"""I will rate the User Answer in comparison to the Reference Answer for a given Question.
A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.
A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.
A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.
I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).
Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.
Return your response as JSON in this format: {{"rating": X}} where X is 0, 2, or 4.

Question: {safe_query}

Reference Answer: {safe_reference_answer}

User Answer: {safe_user_answer}

Rating: """
