import httpx
import openai

from .base import BaseLLM
import time

class OpenAILLM(BaseLLM):
    def __init__(self, url, key, measure=False, **kwargs):
        http_client = httpx.Client(
            timeout=30.0,
        )
        self.client = openai.OpenAI(
            base_url=url,
            api_key=key,
            http_client=http_client
        )
        super().__init__(**kwargs)
        self.measure = measure
        if self.measure:
            self.total_tokens = 0
            self.total_time = 0

    def _is_available(self) -> bool:
        try:
            self.client.models.retrieve(self._model_name)
            return True
        except Exception as e:
            print(f"Model not available: {e}")
            return False

    def chat(self, messages, **kwargs):
        response_format = kwargs.pop("response_format", None)
        start_time = time.time()

        if response_format:
            response = self.client.chat.completions.parse(
                model=self._model_name,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )

        else:
            response = self.client.chat.completions.create(
                model=self._model_name,
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