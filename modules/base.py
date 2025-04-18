import torch
import torch.nn as nn
from functools import lru_cache
from openai import OpenAI
import time
from prompt_template import base_model_prompt
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import clean_text

quantization_config = BitsAndBytesConfig(load_in_8bit=True)


client = OpenAI(
    # api_key = "sk-fd3wC6DBhwl3UAAf1519B466536d4386A80a6aFcBb4e4932",
    # base_url="https://chat.zju.edu.cn/api/ai/v1"
    # volca
    api_key = "6bbc4112-8880-41f3-95bd-e19822d45191", 
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


class BaseModel(nn.Module):
    def __init__(
        self, 
        base_model_name,
        use_hf_model,
        temperature,
        max_tokens,
        top_p,
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.use_hf_model = use_hf_model

        if self.use_hf_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=None,
                quantization_config=quantization_config
            )
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

            for param in self.model.parameters():
                param.requires_grad = False

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
    def call_hf_model(self, prompt):
        text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=128,
            top_p=self.top_p,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def build_prompt(self, context, question):
        return [
            {"role": "system", "content": base_model_prompt['instructions']},
            {"role": "user", "content": base_model_prompt['input'].format(
                contexts="\n\n".join(context),
                question=question
            )}
        ]

    def forward(self, context, question):
        prompt = self.build_prompt(context, question)
        if self.use_hf_model:
            generated_answer = clean_text(self.call_hf_model(prompt))
        else:
            # convert to json string for lru cache
            generated_answer = clean_text(self.call_api(json.dumps(prompt)))
        
        return generated_answer
