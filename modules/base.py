import torch.nn as nn
from functools import lru_cache
from openai import OpenAI
import time
from prompt_template import base_model_prompt
import json


client = OpenAI(
    api_key = "sk-fd3wC6DBhwl3UAAf1519B466536d4386A80a6aFcBb4e4932",
    base_url="https://chat.zju.edu.cn/api/ai/v1"
)


class BaseModel(nn.Module):
    def __init__(
        self, 
        base_model_name,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    @lru_cache(maxsize=10000)
    def call_api(self, prompt):
        prompt = json.loads(prompt)
        patience = 10
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.base_model_name,
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=["\n"]
                )
                output = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(e)
                patience -= 1
                if not patience:
                    print("!!! running out of patience waiting for OpenAI")
                    break
                else:
                    time.sleep(0.1)

        return output

    def build_prompt(self, context, question):
        return [
            {"role": "system", "content": base_model_prompt['instructions']},
            {"role": "user", "content": base_model_prompt['input'].format(
                contexts="\n\n".join(context),
                question=question
            )}
        ]

    def forward(self, contexts, questions):
        generated_answers = []
        for i in range(len(questions)):
            prompt = self.build_prompt(contexts[i], questions[i])
            # convert to json string for lru cache
            generated_answer = self.call_api(json.dumps(prompt))
            generated_answers.append(generated_answer)
        
        return generated_answers
