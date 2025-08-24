"""
LLM模块
包含大语言模型相关功能和提示词管理
"""

from .base import BaseLLM
from .llm_factory import LLMFactory, llm_factory
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM
from .prompt_manager import PromptManager, prompt_manager
from .transformers_llm import TransformersLLM

__all__ = [
    'BaseLLM',
    'OllamaLLM',
    'OpenAILLM',
    'TransformersLLM',
    'LLMFactory',
    'llm_factory',
    'PromptManager',
    'prompt_manager',
]
