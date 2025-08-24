"""
Prompt configuration management module
Manages all prompt templates using YAML configuration files
"""
import os
from typing import Dict, List, Optional, Any

import yaml


class PromptManager:
    """Prompt manager responsible for loading and managing prompt templates from YAML configuration files"""

    def __init__(self, config_path: str = None):
        """
        Initialize prompt manager

        Args:
            config_path: yaml config file path, defaults to config/prompts.yaml in project root
        """
        if config_path is None:
            # Get project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # Go back to root from src directory
            config_path = os.path.join(project_root, "config", "prompts.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML configuration file parsing error: {e}")

    def get_prompt_template(self, category: str, template_name: str) -> str:
        """
        Get prompt template by category and name

        Args:
            category: Prompt category (e.g., 'error_detection')
            template_name: Template name (e.g., 'general_instruction')

        Returns:
            Prompt template string
        """
        try:
            return self.config['prompts'][category][template_name]
        except KeyError:
            raise KeyError(f"Prompt template not found: {category}.{template_name}")

    def generate_error_detection_prompt(self,
                                        column_name: str,
                                        data_json: str,
                                        label_json: str,
                                        fewshot_examples: Optional[List[Dict]] = None,
                                        rule_content: Optional[str] = None,
                                        include_fewshot: bool = True,
                                        include_rule: bool = True) -> str:
        """
        Generate error detection prompt with configurable components

        Args:
            column_name: Target column name
            data_json: Data JSON string
            label_json: Output template JSON string
            fewshot_examples: Few-shot example list
            rule_content: Rule content
            include_fewshot: Whether to include few-shot examples
            include_rule: Whether to include rules
            exclude_keys:

        Returns:
            Complete error detection prompt
        """

        prompt = self.get_prompt_template('error_detection', 'system_prompt')

        # Start with general instruction (always included)
        general_instruction = self.get_prompt_template('error_detection', 'base_instruction')
        prompt += general_instruction.format(column_name=column_name)

        # Add rule instruction (if enabled and rule_content provided)
        if include_rule and rule_content:
            rule_instruction = self.get_prompt_template('error_detection', 'rule_instruction')
            prompt += rule_instruction.format(column=column_name, rules=rule_content)

        # Add few-shot examples (if enabled and examples provided)
        if include_fewshot and fewshot_examples:
            fewshot_content = self._generate_fewshot_examples(column_name, fewshot_examples)
            fewshot_template = self.get_prompt_template('error_detection', 'fewshot_template')
            prompt += fewshot_template.format(examples=fewshot_content)

        # Add input template (always included)
        input_template = self.get_prompt_template('error_detection', 'input_template')
        prompt += input_template.format(data_json=data_json, column_name=column_name)

        output_template = self.get_prompt_template('error_detection', 'output_template')
        prompt += output_template.format(output_json=label_json)

        return prompt

    def _generate_fewshot_examples(self, column: str, examples: List[Dict], exclude_keys=None) -> str:
        """Generate few-shot example content"""
        if exclude_keys is None:
            exclude_keys = ['clean_value']

        example_template = self.get_prompt_template('error_detection', 'example_template')
        examples_content = ""

        for i, example in enumerate(examples, 1):
            row_data = {k: v for k, v in example.items() if k not in exclude_keys}
            original_value = example[column]
            clean_value = example['clean_value']

            example_content = example_template.format(
                example_num=i,
                row_data=row_data,
                column=column,
                original_value=original_value,
                clean_value=clean_value
            )
            examples_content += example_content + "\n\n"

        return examples_content

    def _generate_rule_prompt(self, column: str, rules: Dict[str, Any]) -> str:
        """
        Generate rule prompt

        Args:
            column: Column name
            rules: Rules dictionary

        Returns:
            Rule prompt string
        """
        if not rules:
            return ""

        rule_template = self.get_prompt_template('rule_generation', 'template')
        rules_content = ""

        for key, value in rules.items():
            rules_content += f"{key}: {value}\n"

        return rule_template.format(column=column, rules=rules_content.strip())

    def reload_config(self):
        """Reload configuration file"""
        self.config = self._load_config()


# Create global prompt manager instance
prompt_manager = PromptManager()
