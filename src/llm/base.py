# base.py
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def chat(self, messages, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt, **kwargs):
        pass
