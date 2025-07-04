import re

import ollama
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def remove_thinking(text: str) -> str:
    """
    Remove <think>...</think> tags from the text.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def is_reasoning_model(model_name: str) -> bool:
    """
    Check if the model is a reasoning model based on its name.
    """
    reasoning_models = ["qwen3", "deepseek-r1", "magistral"]
    return any(model_name.startswith(name) for name in reasoning_models)


class LLMClient:
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize the base model and tokenizer.
        :param model_name: Model name or path
        :param device: 'cuda' or 'cpu', default automatically detects
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Loading model from: {model_name} on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device)

        # Handling eos/pad token
        self.eos_token_id = self.tokenizer.eos_token_id or self.model.config.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.is_peft = False  # Flag indicating whether a PEFT adapter is loaded

    def generate_response(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.2, do_sample: bool = False,
                 repetition_penalty: float = 1.2, top_p: float = 0.90) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input text prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature; higher values produce more diverse outputs.
            do_sample (bool): Whether to use sampling; if False, beam search is used.
            repetition_penalty (float): Penalty for repeating tokens. >1.0 discourages repetition.
            top_p (float): Top-p (nucleus) sampling threshold. Only the most probable tokens with cumulative probability >= top_p are considered.

        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # no_repeat_ngram_size=2,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            # top_p=top_p,
            # do_sample=do_sample,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def estimate_memory_usage(self, verbose: bool = False) -> float:
        """
        Returns the estimated model memory usage (in GB).
        """
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        size_gb = total_size / (1024 ** 3)

        if verbose:
            print(f"[Model Memory Information]")
            print(f"Parameter size: {param_size / (1024 ** 2):.2f} MB")
            print(f"Buffer size: {buffer_size / (1024 ** 2):.2f} MB")
            print(f"Total size: {size_gb:.2f} GB")

        return size_gb

    def load_peft_adapter(self, peft_model_dir: str):
        """
        Loads a PEFT adapter (e.g., LoRA) and injects it into the base model.
        :param peft_model_dir: Path to the PEFT adapter
        """
        try:
            # If it's a standalone adapter directory (without the base model)
            self.model = PeftModel.from_pretrained(self.model, peft_model_dir)
            self.is_peft = True
            print(f"[LoRA] Successfully loaded PEFT adapter from: {peft_model_dir}")
        except Exception:
            # If the PEFT directory contains the full adapter with base model path
            config = PeftConfig.from_pretrained(peft_model_dir)
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(self.device)
            self.model = PeftModel.from_pretrained(base_model, peft_model_dir).to(self.device)
            self.is_peft = True
            print(f"[LoRA] Successfully loaded adapter and reloaded base model from: {peft_model_dir}")

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding for a given text.
        :param text: Input text
        :return: Embedding tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # return outputs.last_hidden_state.mean(dim=1)
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = torch.sum(hidden_states * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_hidden / sum_mask  # [batch, hidden_size]
        return sentence_embeddings


class QwenChatbot:
    def __init__(self, model_name):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Loading model from: {model_name} on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device)
        self.history = []

    def generate(self, user_input, thinking=False):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        print(text)

        inputs = self.tokenizer(text, return_tensors="pt")
        length = len(inputs.input_ids[0])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.generate(**inputs, max_new_tokens=2048).to(self.device)
        response_ids = output[0][length:].tolist()
        # response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response


class OllamaClient:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.supports_reasoning = is_reasoning_model(self.model_name)

    def generate(self, prompt: str, thinking: bool = None, show_thinking: bool = None):
        use_thinking = False
        if self.supports_reasoning:
            use_thinking = thinking if thinking is not None else True

        generate_params = {
            "prompt": prompt,
            "model": self.model_name
        }

        if self.supports_reasoning and use_thinking:
            generate_params["think"] = True


        response = ollama.generate(**generate_params).response

        if self.supports_reasoning and use_thinking and not (show_thinking or (show_thinking is None and use_thinking)):
            response = remove_thinking(response)

        return response

    def generate_stream(self, prompt: str):
        """Stream the chat response."""
        response = ollama.generate(
            prompt=prompt,
            model=self.model_name,
            stream=True
        )

        for chunk in response:
            if chunk.get('response'):
                yield chunk['response']

    def chat(self, messages: list, thinking: bool = True, show_thinking: bool = False):
        use_thinking = False
        if self.supports_reasoning:
            # 对推理模型，默认启用思考模式
            use_thinking = thinking if thinking is not None else True

        # 构建请求参数
        chat_params = {
            "messages": messages,
            "model": self.model_name
        }
        # 仅当模型支持推理且请求思考时添加think参数
        if self.supports_reasoning and use_thinking:
            chat_params["think"] = True

        message = ollama.chat(**chat_params).message

        reasoning = message.thinking if self.supports_reasoning and use_thinking else ""
        response = message.content

        if not show_thinking:
            reasoning = ""

        return reasoning, response

    def chat_stream(self, messages: list):
        """Stream the chat response."""
        response = ollama.chat(
            messages=messages,
            model=self.model_name,
            stream=True
        )

        for chunk in response:
            if chunk.get('message') and chunk['message'].get('content'):
                yield chunk['message']['content']




if __name__ == "__main__":
    # client = OllamaClient(model_name="llama3.1:8b")
    client = LLMClient(model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda")

    prompt = "pneumonia patients given initial antibiotic(s) within 6 hours after arrival"
    # response = client.generate(prompt)
    # print(response)
    embedding = client.get_embedding(prompt)
    print(embedding)
    print(embedding.shape)