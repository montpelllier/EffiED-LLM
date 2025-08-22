"""
LLM模块
包含大语言模型相关功能和提示词管理
"""

from .prompt_manager import PromptManager, prompt_manager

__all__ = [
    'PromptManager',
    'prompt_manager',
]
