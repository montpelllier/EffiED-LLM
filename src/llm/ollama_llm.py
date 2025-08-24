import ollama

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "llama3.1:8b"):
        super().__init__(model_name)

    def _is_available(self) -> bool:
        models_response = ollama.list()
        model_names = [model.model for model in models_response.get("models", [])]
        return self._model_name in model_names

    def generate(self, prompt, **kwargs):
        response = ollama.generate(
            model=self._model_name,
            prompt=prompt,
            **kwargs,
        ).response

        return response

    def chat(self, messages, **kwargs):
        response = ollama.chat(
            model=self._model_name,
            messages=messages,
            **kwargs,
        ).message

        return response.content
