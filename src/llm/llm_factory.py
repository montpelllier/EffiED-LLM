"""
LLM工厂类
提供统一的LLM实例创建接口，支持从YAML配置文件管理模型配置
"""
import os
import yaml
from typing import Dict, Any, Optional, Union

from .base import BaseLLM
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM
from .transformers_llm import TransformersLLM


class LLMFactory:
    """LLM工厂类，用于创建不同类型的LLM实例"""

    # 支持的模型提供商
    SUPPORTED_API = {
        'ollama': OllamaLLM,
        'openai': OpenAILLM,
        'transformers': TransformersLLM
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化LLM工厂

        Args:
            config_path: 配置文件路径，默认为项目根目录下的 config/llm_models.yaml
        """
        if config_path is None:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "config", "llm_models.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML配置文件解析错误: {e}")

    @classmethod
    def create_llm(cls, api: str, model_name: str, **kwargs) -> BaseLLM:
        """
        创建LLM实例（类方法，用于直接创建）

        Args:
            api: 提供商名称 ('ollama', 'openai', 'transformers')
            model_name: 模型名称
            **kwargs: 其他参数

        Returns:
            BaseLLM实例

        Raises:
            ValueError: 不支持的提供商
        """
        if api not in cls.SUPPORTED_API:
            raise ValueError(f"不支持的提供商: {api}. 支持的提供商: {list(cls.SUPPORTED_API.keys())}")

        llm_class = cls.SUPPORTED_API[api]

        # 为不同提供商设置模型名称
        if api == 'ollama':
            return llm_class(model_name=model_name, **kwargs)
        elif api == 'openai':
            # OpenAI需要url和key参数
            if 'url' not in kwargs or 'key' not in kwargs:
                raise ValueError("OpenAI模型需要提供 'url' 和 'key' 参数")
            return llm_class(model_name=model_name, **kwargs)
        elif api == 'transformers':
            # Transformers需要device参数
            device = kwargs.get('device', 'auto')
            return llm_class(model_name=model_name, device=device, **kwargs)
        else:
            raise ValueError("")

    def create_llm_from_config(self, model_name: str, **override_kwargs) -> BaseLLM:
        """
        从配置文件创建LLM实例

        Args:
            model_name: 配置文件中的模型名称
            **override_kwargs: 覆盖配置的参数

        Returns:
            BaseLLM实例

        Raises:
            KeyError: 模型不存在于配置中
            ValueError: 配置错误
        """
        if model_name not in self.config['models']:
            raise KeyError(f"模型 '{model_name}' 不存在于配置中. 可用模型: {list(self.config['models'].keys())}")

        model_config = self.config['models'][model_name].copy()
        api = model_config.pop('api')

        # 如果模型配置中指定了 api_config，从 api_configs 中获取 URL 和 key
        api_config_name = model_config.pop('api_config', None)
        if api_config_name and 'api_configs' in self.config:
            api_config = self.config['api_configs'].get(api_config_name, {})
            # 如果模型配置中没有指定 url 或 key，从 api_configs 中获取
            if 'url' not in model_config and 'url' in api_config:
                model_config['url'] = api_config['url']
            if 'key' not in model_config and 'key' in api_config:
                model_config['key'] = api_config['key']

        # 合并参数: 默认参数 < 模型配置 < 覆盖参数
        final_kwargs = {}
        final_kwargs.update(model_config)
        final_kwargs.update(override_kwargs)

        return self.create_llm(api, model_name, **final_kwargs)

    def get_api_config(self, api_name: str) -> Dict[str, str]:
        """
        获取 API 配置

        Args:
            api_name: API 配置名称 (如 'google', 'openai', 'openrouter')

        Returns:
            API 配置字典
        """
        return self.config.get('api_configs', {}).get(api_name, {})

    def get_available_models(self) -> Dict[str, str]:
        """
        获取可用模型列表

        Returns:
            模型名称到提供商的映射
        """
        return {model: config['api'] for model, config in self.config['models'].items()}

    def get_models_by_api(self, api: str) -> list:
        """
        获取指定提供商的模型列表

        Args:
            provider: 提供商名称

        Returns:
            模型名称列表
        """
        return [model for model, config in self.config['models'].items()
                if config['api'] == api]

    def reload_config(self):
        """重新加载配置文件"""
        self.config = self._load_config()



# 创建全局工厂实例
llm_factory = LLMFactory()
