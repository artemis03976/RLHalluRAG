import torch.nn as nn
from functools import lru_cache
from openai import OpenAI
import time
from prompt_template import evaluator_prompt
from utils import extract_judgement
import json


client = OpenAI(
    # api_key = "sk-fd3wC6DBhwl3UAAf1519B466536d4386A80a6aFcBb4e4932",
    # base_url="https://chat.zju.edu.cn/api/ai/v1",
    # volca
    api_key = "6bbc4112-8880-41f3-95bd-e19822d45191", 
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


class HalluEvaluator(nn.Module):
    def __init__(
        self, 
        evaluator_name,
        temperature,
        max_tokens,
        top_p,
    ) -> None:
        super().__init__()

        self.evaluator_name = evaluator_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

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

    def forward(self, question, golden_answer, generated_answer):
        # generate the prompt input
        prompt = self.build_evaluator_prompt(question, golden_answer, generated_answer)
        # get the output from evaluator model
        output = self.call_api(json.dumps(prompt))
        # extract the judgement from the output
        judgement = extract_judgement(output)

        return judgement
