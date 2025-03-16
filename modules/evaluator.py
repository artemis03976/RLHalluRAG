import torch.nn as nn
from functools import lru_cache
from openai import OpenAI
import time
from prompt_template import evaluator_prompt
from utils import extract_judgement
import json


client = OpenAI(
    api_key = "sk-fd3wC6DBhwl3UAAf1519B466536d4386A80a6aFcBb4e4932",
    base_url="https://chat.zju.edu.cn/api/ai/v1"
)



class HalluEvaluator(nn.Module):
    def __init__(
        self, 
        evaluator_name,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty
    ) -> None:
        super().__init__()

        self.evaluator_name = evaluator_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    @lru_cache(maxsize=10000)
    def call_api(self, prompt):
        prompt = json.loads(prompt)
        patience = 100
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.evaluator_name,
                    messages=prompt,
                    temperature=0.0,
                    max_tokens=32,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=["\n"]
                )
                output = response.choices[0].message.content.strip()
                break
            except Exception as e:
                patience -= 1
                if not patience:
                    print("!!! running out of patience waiting for OpenAI")
                    break
                else:
                    time.sleep(0.1)
        return output
    
    def build_evaluator_prompt(self, question, golden_answers, generated_answer):
        return [
            {"role": "system", "content": evaluator_prompt['instructions']},
            {"role": "user", "content": evaluator_prompt['input'].format(
                question=question, 
                golden_answers=golden_answers, 
                generated_answer=generated_answer
            )}
        ]

    def forward(self, questions, golden_answers, generated_answers):
        judgements = []

        # loop over the training examples
        for i in range(len(questions)):
            # generate the prompt input
            prompt = self.build_evaluator_prompt(questions[i], golden_answers[i], generated_answers[i])
            # get the output from evaluator model
            output = self.call_api(json.dumps(prompt))
            # extract the judgement from the output
            judgement = extract_judgement(output)
            judgements.append(judgement)

        return judgements
