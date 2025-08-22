import ollama

from .base import BaseLLM


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