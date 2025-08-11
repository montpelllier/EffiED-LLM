"""
测试detection模块中的prompt_manager功能
"""
import unittest
import sys
from pathlib import Path

# 添加路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from detection.prompt_manager import PromptManager


class TestPromptManager(unittest.TestCase):
    """测试PromptManager类"""

    def setUp(self):
        """设置测试环境"""
        self.prompt_manager = PromptManager()

        # 测试数据
        self.test_data_values = ['apple', 'banana', 'invalid123', 'orange', 'xyz']
        self.test_column_name = 'fruit_name'
        self.test_column_rule = {
            'name': 'fruit_name',
            'meaning': 'Name of fruit',
            'data_type': 'string',
            'format_rule': 'fruit names only',
            'null_value_rule': 'not allowed'
        }

        self.test_examples = [
            {
                'value': 'apple',
                'label': 0,
                'explanation': 'Valid fruit name'
            },
            {
                'value': 'invalid123',
                'label': 1,
                'explanation': 'Contains numbers, not a valid fruit name'
            }
        ]

    def test_load_default_templates(self):
        """测试默认模板加载"""
        templates = self.prompt_manager._load_default_templates()

        expected_templates = ['zero_shot_base', 'few_shot_base', 'rule_based']
        for template_name in expected_templates:
            self.assertIn(template_name, templates)
            self.assertIsInstance(templates[template_name], str)
            self.assertGreater(len(templates[template_name]), 0)

    def test_generate_zero_shot_prompt(self):
        """测试zero-shot提示词生成"""
        prompt = self.prompt_manager.generate_zero_shot_prompt(
            self.test_data_values,
            self.test_column_name,
            self.test_column_rule
        )

        # 检查提示词包含必要信息
        self.assertIn(self.test_column_name, prompt)
        self.assertIn('apple', prompt)
        self.assertIn('banana', prompt)
        self.assertIn('JSON', prompt)
        self.assertIn('labels', prompt)
        self.assertIn('Name of fruit', prompt)  # column meaning

    def test_generate_few_shot_prompt(self):
        """测试few-shot提示词生成"""
        prompt = self.prompt_manager.generate_few_shot_prompt(
            self.test_data_values,
            self.test_column_name,
            self.test_column_rule,
            self.test_examples
        )

        # 检查提示词包含必要信息
        self.assertIn(self.test_column_name, prompt)
        self.assertIn('apple', prompt)  # 数据值
        self.assertIn('examples', prompt.lower())
        self.assertIn('Valid fruit name', prompt)  # example explanation
        self.assertIn('JSON', prompt)

    def test_generate_rule_based_prompt(self):
        """测试基于规则的提示词生成"""
        prompt = self.prompt_manager.generate_rule_based_prompt(
            self.test_data_values,
            self.test_column_name,
            self.test_column_rule
        )

        # 检查提示词包含规则信息
        self.assertIn(self.test_column_name, prompt)
        self.assertIn('apple', prompt)
        self.assertIn('Data type must be', prompt)
        self.assertIn('string', prompt)
        self.assertIn('Null values are not allowed', prompt)

    def test_format_column_info(self):
        """测试列信息格式化"""
        column_info = self.prompt_manager._format_column_info(self.test_column_rule)

        self.assertIn('Meaning:', column_info)
        self.assertIn('Name of fruit', column_info)
        self.assertIn('Data Type:', column_info)
        self.assertIn('string', column_info)

    def test_format_data_values(self):
        """测试数据值格式化"""
        formatted = self.prompt_manager._format_data_values(self.test_data_values)

        self.assertIn('1. apple', formatted)
        self.assertIn('2. banana', formatted)
        self.assertIn('3. invalid123', formatted)
        self.assertIn('4. orange', formatted)
        self.assertIn('5. xyz', formatted)

        # 检查格式正确
        lines = formatted.split('\n')
        self.assertEqual(len(lines), 5)

    def test_format_examples(self):
        """测试示例格式化"""
        formatted = self.prompt_manager._format_examples(self.test_examples)

        self.assertIn('Example 1:', formatted)
        self.assertIn('apple', formatted)
        self.assertIn('Correct', formatted)
        self.assertIn('Example 2:', formatted)
        self.assertIn('invalid123', formatted)
        self.assertIn('Error', formatted)
        self.assertIn('Valid fruit name', formatted)
        self.assertIn('Contains numbers', formatted)

    def test_format_rules(self):
        """测试规则格式化"""
        formatted = self.prompt_manager._format_rules(self.test_column_rule)

        self.assertIn('Data type must be: string', formatted)
        self.assertIn('Format rule: fruit names only', formatted)
        self.assertIn('Null values are not allowed', formatted)

        # 检查每条规则都在单独行
        lines = formatted.split('\n')
        self.assertGreaterEqual(len(lines), 3)

    def test_format_rules_empty(self):
        """测试空规则格式化"""
        formatted = self.prompt_manager._format_rules(None)
        self.assertEqual(formatted, "No specific rules provided.")

        formatted_empty = self.prompt_manager._format_rules({})
        self.assertEqual(formatted_empty, "No specific rules provided.")

    def test_add_custom_template(self):
        """测试添加自定义模板"""
        custom_template = "This is a custom template for {column_name}"
        template_name = "custom_test"

        self.prompt_manager.add_custom_template(template_name, custom_template)

        # 验证模板已添加
        self.assertIn(template_name, self.prompt_manager.templates)
        self.assertEqual(self.prompt_manager.templates[template_name], custom_template)

    def test_get_template(self):
        """测试获取模板"""
        template = self.prompt_manager.get_template('zero_shot_base')
        self.assertIsInstance(template, str)
        self.assertGreater(len(template), 0)

        # 测试获取不存在的模板
        empty_template = self.prompt_manager.get_template('nonexistent')
        self.assertEqual(empty_template, "")

    def test_list_templates(self):
        """测试列出所有模板"""
        templates = self.prompt_manager.list_templates()

        self.assertIsInstance(templates, list)
        self.assertIn('zero_shot_base', templates)
        self.assertIn('few_shot_base', templates)
        self.assertIn('rule_based', templates)

    def test_prompt_consistency(self):
        """测试提示词一致性"""
        # 相同输入应该生成相同提示词
        prompt1 = self.prompt_manager.generate_zero_shot_prompt(
            self.test_data_values,
            self.test_column_name,
            self.test_column_rule
        )

        prompt2 = self.prompt_manager.generate_zero_shot_prompt(
            self.test_data_values,
            self.test_column_name,
            self.test_column_rule
        )

        self.assertEqual(prompt1, prompt2)

    def test_prompt_structure(self):
        """测试提示词结构完整性"""
        prompts = {
            'zero_shot': self.prompt_manager.generate_zero_shot_prompt(
                self.test_data_values, self.test_column_name, self.test_column_rule
            ),
            'few_shot': self.prompt_manager.generate_few_shot_prompt(
                self.test_data_values, self.test_column_name,
                self.test_column_rule, self.test_examples
            ),
            'rule_based': self.prompt_manager.generate_rule_based_prompt(
                self.test_data_values, self.test_column_name, self.test_column_rule
            )
        }

        for prompt_type, prompt in prompts.items():
            with self.subTest(prompt_type=prompt_type):
                # 所有提示词都应该包含这些基本元素
                self.assertIn(self.test_column_name, prompt)
                self.assertIn('JSON', prompt)
                self.assertIn('labels', prompt)
                self.assertIn('[0, 1, 0, 1, ...]', prompt)
                self.assertIn('Return only the JSON object', prompt)

    def test_edge_cases(self):
        """测试边界情况"""
        # 空数据值
        prompt_empty = self.prompt_manager.generate_zero_shot_prompt(
            [], self.test_column_name, self.test_column_rule
        )
        self.assertIn(self.test_column_name, prompt_empty)

        # 无规则信息
        prompt_no_rule = self.prompt_manager.generate_zero_shot_prompt(
            self.test_data_values, self.test_column_name, None
        )
        self.assertIn(self.test_column_name, prompt_no_rule)

        # 空示例
        prompt_no_examples = self.prompt_manager.generate_few_shot_prompt(
            self.test_data_values, self.test_column_name,
            self.test_column_rule, None
        )
        self.assertIn(self.test_column_name, prompt_no_examples)


if __name__ == '__main__':
    print("=== 运行detection.prompt_manager测试 ===")
    unittest.main(verbosity=2)
