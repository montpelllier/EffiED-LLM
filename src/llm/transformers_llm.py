import torch

from llm.base import BaseLLM


class TransformersLLM(BaseLLM):
    def __init__(self, model_name, device):
        super().__init__(model_name=model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Loading model from: {model_name} on device: {self.device}")

    def _is_available(self) -> bool:
        pass

    def chat(self, messages, **kwargs):
        raise NotImplementedError("Not implemented.")

    def generate(self, prompt, **kwargs):
        raise NotImplementedError("Not implemented.")
