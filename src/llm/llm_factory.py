"""
LLM工厂类
提供统一的LLM实例创建接口
"""
from typing import Dict, Any, Optional
from .base_llm import BaseLLM
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM


class LLMFactory:
    """LLM工厂类，用于创建不同类型的LLM实例"""

    # 支持的模型类型
    SUPPORTED_PROVIDERS = {
        'ollama': OllamaLLM,
        'openai': OpenAILLM
    }

    # 常用模型配置
    COMMON_MODELS = {
        # Ollama模型
        'llama3.1:8b': {'provider': 'ollama'},
        'llama3.2:3b': {'provider': 'ollama'},
        'qwen2.5:7b': {'provider': 'ollama'},
        'qwen3:4b': {'provider': 'ollama'},
        'qwen3:8b': {'provider': 'ollama'},
        'gemma3:4b': {'provider': 'ollama'},

        # OpenAI模型
        'gpt-3.5-turbo': {'provider': 'openai'},
        'gpt-4': {'provider': 'openai'},
        'gpt-4-turbo': {'provider': 'openai'}
    }

    @classmethod
    def create_llm(cls, provider: str, model_name: str, **kwargs) -> BaseLLM:
        """
        创建LLM实例

        Args:
            provider: 提供商类型 ('ollama' 或 'openai')
            model_name: 模型名称
            **kwargs: 额外配置参数

        Returns:
            LLM实例
        """
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported providers: {list(cls.SUPPORTED_PROVIDERS.keys())}")

        llm_class = cls.SUPPORTED_PROVIDERS[provider]
        return llm_class(model_name=model_name, **kwargs)

    @classmethod
    def create_ollama_llm(cls, model_name: str, host: str = None, **kwargs) -> OllamaLLM:
        """
        创建Ollama LLM实例
        """
        return cls.create_llm('ollama', model_name, host=host, **kwargs)

    @classmethod
    def create_openai_llm(cls, model_name: str, api_key: str,
                         base_url: str = None, **kwargs) -> OpenAILLM:
        """
        创建OpenAI LLM实例
        """
        return cls.create_llm('openai', model_name, api_key=api_key,
                            base_url=base_url, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseLLM:
        """
        从配置字典创建LLM实例

        Args:
            config: 配置字典，必须包含 'provider' 和 'model_name'
        """
        config = config.copy()
        provider = config.pop('provider')
        model_name = config.pop('model_name')

        return cls.create_llm(provider, model_name, **config)

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取预定义模型的信息
        """
        return cls.COMMON_MODELS.get(model_name)

    @classmethod
    def list_supported_providers(cls) -> list:
        """
        获取支持的提供商列表
        """
        return list(cls.SUPPORTED_PROVIDERS.keys())

    @classmethod
    def list_common_models(cls) -> list:
        """
        获取常用模型列表
        """
        return list(cls.COMMON_MODELS.keys())
