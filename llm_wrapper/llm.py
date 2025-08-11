# ollama_llm.py
import time

import ollama
import torch
from openai import OpenAI

from llm_wrapper.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, prompt, **kwargs):
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            **kwargs,
        ).response

        return response

    def chat(self, messages, **kwargs):
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            **kwargs,
        ).message

        return response.content


class OpenAILLM(BaseLLM):
    def __init__(self, url, key, measure=False, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            base_url=url,
            api_key=key,
        )
        self.measure = measure
        if self.measure:
            self.total_tokens = 0
            self.total_time = 0

    def chat(self, messages, **kwargs):
        response_format = kwargs.pop("response_format", None)
        start_time = time.time()

        if response_format:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )

        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
        if self.measure:
            usage_info = response.usage
            print(usage_info)
            self.total_tokens += usage_info.total_tokens
            self.total_time += time.time() - start_time

        return response.choices[0].message.content

    def generate(self, prompt, **kwargs):
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


class TransformersLLM(BaseLLM):
    def __init__(self, model_name, device):
        super().__init__(model_name=model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Loading model from: {model_name} on device: {self.device}")

    def chat(self, messages, **kwargs):
        raise NotImplementedError("TransformersLLM does not support chat method.")

    def generate(self, prompt, **kwargs):
        raise NotImplementedError("TransformersLLM does not support generate method.")
