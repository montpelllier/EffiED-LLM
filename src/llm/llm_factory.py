import os
from typing import Dict, Any, Optional

import yaml

from .base import BaseLLM
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM
from .transformers_llm import TransformersLLM


class LLMFactory:
    SUPPORTED_API = {
        'ollama': OllamaLLM,
        'openai': OpenAILLM,
        'transformers': TransformersLLM
    }

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Get project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "config", "llm_models.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML configuration file parsing error: {e}")

    @classmethod
    def create_llm(cls, api: str, model_name: str, **kwargs) -> BaseLLM:
        """
        Create LLM instance (class method for direct creation)

        Args:
            api: api name ('ollama', 'openai', 'transformers')
            model_name: model name
            **kwargs: other parameters

        Returns:
            BaseLLM instance

        """
        if api not in cls.SUPPORTED_API:
            raise ValueError(f"Unsupported API: {api}. Supported values: {list(cls.SUPPORTED_API.keys())}")

        llm_class = cls.SUPPORTED_API[api]

        if api == 'ollama':
            return llm_class(model_name=model_name, **kwargs)
        elif api == 'openai':
            if 'url' not in kwargs or 'key' not in kwargs:
                raise ValueError("OpenAI model requires 'url' and 'key' parameters")
            return llm_class(model_name=model_name, **kwargs)
        elif api == 'transformers':
            device = kwargs.get('device', 'auto')
            return llm_class(model_name=model_name, device=device, **kwargs)
        else:
            raise ValueError("")

    def create_llm_from_config(self, model_name: str, **override_kwargs) -> BaseLLM:
        """
        Create LLM instance from configuration file

        Args:
            model_name: model name in configuration file
            **override_kwargs: parameters to override configuration

        Returns:
            BaseLLM instance

        Raises:
            KeyError: model does not exist in configuration
            ValueError: configuration error
        """
        if model_name not in self.config['models']:
            raise KeyError(
                f"Model '{model_name}' does not exist in configuration. Available models: {list(self.config['models'].keys())}")

        model_config = self.config['models'][model_name].copy()
        api = model_config.pop('api')

        # If api_config is specified in model configuration, get URL and key from api_configs
        api_config_name = model_config.pop('api_config', None)
        if api_config_name and 'api_configs' in self.config:
            api_config = self.config['api_configs'].get(api_config_name, {})
            # If url or key is not specified in model config, get from api_configs
            if 'url' not in model_config and 'url' in api_config:
                model_config['url'] = api_config['url']
            if 'key' not in model_config and 'key' in api_config:
                model_config['key'] = api_config['key']

        # Merge parameters: default parameters < model config < override parameters
        final_kwargs = {}
        final_kwargs.update(model_config)
        final_kwargs.update(override_kwargs)

        return self.create_llm(api, model_name, **final_kwargs)

    def get_api_config(self, api_name: str) -> Dict[str, str]:
        """
        Get API configuration

        Args:
            api_name: API configuration name (e.g., 'google', 'openai', 'openrouter')

        Returns:
            API configuration dictionary
        """
        return self.config.get('api_configs', {}).get(api_name, {})

    def get_available_models(self) -> Dict[str, str]:
        """
        Get available models list

        Returns:
            Mapping from model name to provider
        """
        return {model: config['api'] for model, config in self.config['models'].items()}

    def get_models_by_api(self, api: str) -> list:
        """
        Get model list for specified provider

        Args:
            api: provider name

        Returns:
            List of model names
        """
        return [model for model, config in self.config['models'].items()
                if config['api'] == api]

    def reload_config(self):
        """Reload configuration file"""
        self.config = self._load_config()


# Create global factory instance
llm_factory = LLMFactory()
