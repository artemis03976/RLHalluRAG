base_model_prompt = {
    "instructions": (
        'Here is a question and some relevant contexts(separated by “\\n\\n”). Please provide an answer to the question based on the given infomations.\n'
        'If the question is not answerable based on the given infomations, please answer "this question cannot be answered".'
    ),
    "input": (
        'Contexts: {contexts}\n\n'
        'Question: {question}\n'
        'Answer:'
    )
}

evaluator_prompt = {
    "instructions": (
        'Here is a question, a set of golden answers (separated by “/”), and an AI-generated answer. ' 
        'Your task is to determine whether the AI-generated answer contains any hallucinations relative to the question and the golden answers. '
        'Apply the following criteria:\n'
        '1. Language Fluency: If the AI-generated answer is not fluent (e.g., contains garbled text or excessive code), judge it as hallucinated;\n'
        '2. Relevance: If the answer does not directly address the question, judge it as hallucinated;\n'
        '3. Consistency: If the answer includes information that contradicts any of the golden answers, judge it as hallucinated;\n'
        '4. Semantic Alignment: If the overall meaning of the answer aligns with the golden answers, it should be considered as not hallucinated.;\n'
        '5. Unanswerable Cases: If the golden answers state that “this question cannot be answered” and the AI-generated answer reflects the same, judge it as not hallucinated.\n'
        'Based on these guidelines, please reply with only a single word: “Yes” if the AI-generated answer contains hallucinations, or “No” if it does not.'
    ),
    "input": (
        'Question: {question}\n'
        'Golden Answers: {golden_answers}\n'
        'AI-generated answer: {generated_answer}\n'
        'Judgement:'
    )
}