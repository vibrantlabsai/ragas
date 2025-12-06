"""Context Recall prompt for classifying statement attributions."""

import json


def context_recall_prompt(question: str, context: str, answer: str) -> str:
    """
    Generate the prompt for context recall evaluation.

    Args:
        question: The original question
        context: The retrieved context to evaluate against
        answer: The reference answer containing statements to classify

    Returns:
        Formatted prompt string for the LLM
    """
    # Use json.dumps() to safely escape the strings
    safe_question = json.dumps(question)
    safe_context = json.dumps(context)
    safe_answer = json.dumps(answer)

    return f"""Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason.

--------EXAMPLES-----------
Example 1
Input: {{
    "question": "What can you tell me about Albert Einstein?",
    "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
    "answer": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895."
}}
Output: {{
    "classifications": [
        {{
            "statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
            "reason": "The date of birth of Einstein is mentioned clearly in the context.",
            "attributed": 1
        }},
        {{
            "statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
            "reason": "The exact sentence is present in the given context.",
            "attributed": 1
        }},
        {{
            "statement": "He published 4 papers in 1905.",
            "reason": "There is no mention about papers he wrote in the given context.",
            "attributed": 0
        }},
        {{
            "statement": "Einstein moved to Switzerland in 1895.",
            "reason": "There is no supporting evidence for this in the given context.",
            "attributed": 0
        }}
    ]
}}
-----------------------------

Now perform the same with the following input
Input: {{
    "question": {safe_question},
    "context": {safe_context},
    "answer": {safe_answer}
}}
Output: """
