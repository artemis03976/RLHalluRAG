base_model_prompt = {
    "instructions": (
        'You are given a question and a set of contexts which may or may not be relevant to the question. '
        'If the contexts contain information relevant to the question, answer based strictly on the contexts. '
        'If the contexts do not contain relevant information, use your own knowledge to answer the question.\n'
    ),
    "input": (
        'Contexts: {contexts}\n\n'
        'Question: {question}\n'
        'Answer:'
    )
}

# evaluator_prompt = {
#     "instructions": (
#         'Here is a question, a set of golden answers, and an AI-generated answer. ' 
#         'Your task is to determine whether the AI-generated answer contains any hallucinations relative to the question and the golden answers. '
#         'Apply the following criteria:\n'
#         '1. Language Fluency: If the AI-generated answer is not fluent, judge it as hallucinated;\n'
#         '2. Relevance: If the answer does not directly address the question, judge it as hallucinated;\n'
#         '3. Consistency: If the answer includes information that contradicts any of the golden answers, judge it as hallucinated;\n'
#         '4. Semantic Alignment: If the overall meaning of the answer aligns with the golden answers, it should be considered as not hallucinated.;\n'
#         '5. Unanswerable Cases: If the golden answers state that “this question cannot be answered” and the AI-generated answer reflects the same, judge it as not hallucinated.\n'
#         'Based on these guidelines, please reply with only a single word: “Yes” if the AI-generated answer contains hallucinations, or “No” if it does not.'
#     ),
#     "input": (
#         'Question: {question}\n'
#         'Golden Answers: {golden_answers}\n'
#         'AI-generated answer: {generated_answer}\n'
#         'Judgement:'
#     )
# }

evaluator_prompt = {
    "instructions": (
        "Assess if the AI answer contains hallucinations against the question and golden answers.\n"
        "Criteria:\n"
        "1. Non-fluent → Yes\n"
        "2. Irrelevant → Yes\n"
        "3. Contradicts gold → Yes\n"
        "4. Semantically aligns → No\n"
        "5. Both unanswerable → No\n"
        "Reply only 'Yes' or 'No'"
    ),
    "input": (
        "Q: {question}\n"
        "GA: {golden_answers}\n"
        "AI: {generated_answer}\n"
        "Judgement:"
    )
}