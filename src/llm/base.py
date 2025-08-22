# base.py
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, model_name):
        self._model_name = model_name
        if not self._is_available():
            raise ValueError(f"Model {model_name} not available.")

    def get_model_name(self):
        return self._model_name

    @abstractmethod
    def _is_available(self) -> bool:
        pass

    @abstractmethod
    def chat(self, messages, **kwargs) -> str:
        pass

    @abstractmethod
    def generate(self, prompt, **kwargs) -> str:
        pass
