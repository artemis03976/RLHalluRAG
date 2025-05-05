import torch
import torch.nn as nn
from functools import lru_cache
import time
from prompt_template import base_model_prompt
import json
from utils import clean_text
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# client = OpenAI()
client = None

class BaseModel(nn.Module):
    def __init__(
        self, 
        base_model_name,
        use_hf_model,
        temperature,
        max_tokens,
        top_p,
        n_gpus,
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.use_hf_model = use_hf_model

        if self.use_hf_model:
            self.model = LLM(
                model=base_model_name,
                tokenizer=base_model_name,
                tensor_parallel_size=n_gpus,
                trust_remote_code=True,
                enforce_eager=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                skip_special_tokens=True
            )

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
    
    @torch.inference_mode()
    def call_hf_model(self, batch_prompts):
        outputs = self.model.generate(batch_prompts, self.sampling_params)

        return [clean_text(o.outputs[0].text) for o in outputs]

    def build_prompt(self, context, question):
        messages = [
            {"role": "system", "content": base_model_prompt['instructions']},
            {"role": "user", "content": base_model_prompt['input'].format(
                contexts="\n\n".join(context),
                question=question
            )}
        ]
        
        if self.use_hf_model:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return messages

    def forward(self, shot_contexts, questions):
        batch_prompts = [self.build_prompt(ctx, questions) for ctx in shot_contexts]
        if self.use_hf_model:
            generated_answers = self.call_hf_model(batch_prompts)
        else:
            # convert to json string for lru cache
            generated_answers = clean_text(self.call_api(json.dumps(batch_prompts)))
        
        return generated_answers
