# ollama_llm.py
import ollama
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
    def __init__(self, url, key, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            base_url=url,
            api_key=key,
        )

    def chat(self, messages, **kwargs):
        response_format = kwargs.pop("response_format", None)

        if response_format:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                **kwargs,
            ).choices[0].message.parsed

        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            ).choices[0].message.content

        return response

    def generate(self, prompt, **kwargs):
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
