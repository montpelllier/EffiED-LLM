"""
提示词管理器
管理各种错误检测的提示词模板
"""
import json
from typing import List, Dict, Any, Optional


class PromptManager:
    """提示词管理器，生成各种检测场景的提示词"""

    def __init__(self):
        self.templates = self._load_default_templates()

    def _load_default_templates(self) -> Dict[str, str]:
        """加载默认提示词模板"""
        return {
            'zero_shot_base': """You are a data quality expert. Please identify errors in the following data values for column "{column_name}".

Column: {column_name}
{column_info}

Data values to check:
{data_values}

Please return your analysis as a JSON object with the following format:
{{"labels": [0, 1, 0, 1, ...]}}

Where 0 means correct/clean data and 1 means incorrect/dirty data.
Return only the JSON object, no additional text.""",

            'few_shot_base': """You are a data quality expert. Please identify errors in the following data values for column "{column_name}".

Column: {column_name}
{column_info}

Here are some examples:
{examples}

Now, please check these data values:
{data_values}

Please return your analysis as a JSON object with the following format:
{{"labels": [0, 1, 0, 1, ...]}}

Where 0 means correct/clean data and 1 means incorrect/dirty data.
Return only the JSON object, no additional text.""",

            'rule_based': """You are a data quality expert. Please identify errors in the following data values for column "{column_name}" based on the specified rules.

Column: {column_name}
{column_info}

Rules to check:
{rules_text}

Data values to check:
{data_values}

Please return your analysis as a JSON object with the following format:
{{"labels": [0, 1, 0, 1, ...]}}

Where 0 means correct/clean data and 1 means incorrect/dirty data.
Return only the JSON object, no additional text."""
        }

    def generate_zero_shot_prompt(self, data_values: List, column_name: str,
                                column_rule: Dict[str, Any] = None) -> str:
        """生成zero-shot提示词"""
        column_info = self._format_column_info(column_rule) if column_rule else ""
        data_values_text = self._format_data_values(data_values)

        return self.templates['zero_shot_base'].format(
            column_name=column_name,
            column_info=column_info,
            data_values=data_values_text
        )

    def generate_few_shot_prompt(self, data_values: List, column_name: str,
                               column_rule: Dict[str, Any] = None,
                               examples: List[Dict] = None) -> str:
        """生成few-shot提示词"""
        column_info = self._format_column_info(column_rule) if column_rule else ""
        data_values_text = self._format_data_values(data_values)
        examples_text = self._format_examples(examples) if examples else ""

        return self.templates['few_shot_base'].format(
            column_name=column_name,
            column_info=column_info,
            examples=examples_text,
            data_values=data_values_text
        )

    def generate_rule_based_prompt(self, data_values: List, column_name: str,
                                 column_rule: Dict[str, Any] = None) -> str:
        """生成基于规则的提示词"""
        column_info = self._format_column_info(column_rule) if column_rule else ""
        data_values_text = self._format_data_values(data_values)
        rules_text = self._format_rules(column_rule) if column_rule else "No specific rules provided."

        return self.templates['rule_based'].format(
            column_name=column_name,
            column_info=column_info,
            rules_text=rules_text,
            data_values=data_values_text
        )

    def _format_column_info(self, column_rule: Dict[str, Any]) -> str:
        """格式化列信息"""
        if not column_rule:
            return ""

        info_parts = []

        if 'meaning' in column_rule:
            info_parts.append(f"Meaning: {column_rule['meaning']}")

        if 'data_type' in column_rule:
            info_parts.append(f"Data Type: {column_rule['data_type']}")

        return "\n".join(info_parts)

    def _format_data_values(self, data_values: List) -> str:
        """格式化数据值"""
        formatted_values = []
        for i, value in enumerate(data_values):
            formatted_values.append(f"{i+1}. {value}")
        return "\n".join(formatted_values)

    def _format_examples(self, examples: List[Dict]) -> str:
        """格式化示例"""
        if not examples:
            return ""

        example_texts = []
        for i, example in enumerate(examples):
            value = example.get('value', 'N/A')
            label = example.get('label', 0)
            explanation = example.get('explanation', '')

            example_text = f"Example {i+1}: '{value}' -> {'Error' if label == 1 else 'Correct'}"
            if explanation:
                example_text += f" ({explanation})"

            example_texts.append(example_text)

        return "\n".join(example_texts)

    def _format_rules(self, column_rule: Dict[str, Any]) -> str:
        """格式化规则信息"""
        if not column_rule:
            return "No specific rules provided."

        rules = []

        if 'data_type' in column_rule:
            rules.append(f"- Data type must be: {column_rule['data_type']}")

        if 'format_rule' in column_rule and column_rule['format_rule'] != 'no specific format':
            rules.append(f"- Format rule: {column_rule['format_rule']}")

        if 'null_value_rule' in column_rule:
            if column_rule['null_value_rule'] == 'not allowed':
                rules.append("- Null values are not allowed")
            elif column_rule['null_value_rule'] == 'allowed':
                rules.append("- Null values are allowed")

        return "\n".join(rules) if rules else "No specific rules provided."

    def add_custom_template(self, template_name: str, template_content: str):
        """添加自定义模板"""
        self.templates[template_name] = template_content

    def get_template(self, template_name: str) -> str:
        """获取指定模板"""
        return self.templates.get(template_name, "")

    def list_templates(self) -> List[str]:
        """列出所有可用模板"""
        return list(self.templates.keys())
