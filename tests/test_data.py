"""
测试data模块的功能
包括DatasetLoader和DataManager的测试
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import os
import sys

# 添加路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from data import DatasetLoader, DataManager


class TestDatasetLoader(unittest.TestCase):
    """测试DatasetLoader类"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时目录和测试数据
        self.temp_dir = tempfile.mkdtemp()
        self.test_datasets_dir = Path(self.temp_dir) / "datasets"
        self.test_datasets_dir.mkdir()

        # 创建测试数据集
        self.test_dataset_name = "test_dataset"
        dataset_dir = self.test_datasets_dir / self.test_dataset_name
        dataset_dir.mkdir()

        # 创建测试CSV文件
        self.clean_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        self.dirty_data = pd.DataFrame({
            'col1': [1, 2, 999, 4, 5],  # 999是错误值
            'col2': ['a', 'b', 'x', 'd', 'e'],  # x是错误值
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        # 保存测试数据
        self.clean_data.to_csv(dataset_dir / "clean.csv", index=False)
        self.dirty_data.to_csv(dataset_dir / "dirty.csv", index=False)

        # 创建规则文件
        self.test_rules = {
            "columns": [
                {
                    "name": "col1",
                    "meaning": "Test integer column",
                    "data_type": "integer",
                    "format_rule": "positive integer",
                    "null_value_rule": "not allowed"
                },
                {
                    "name": "col2",
                    "meaning": "Test string column",
                    "data_type": "string",
                    "format_rule": "single letter",
                    "null_value_rule": "not allowed"
                }
            ]
        }

        with open(dataset_dir / "rule.json", 'w') as f:
            json.dump(self.test_rules, f)

        self.loader = DatasetLoader(str(self.test_datasets_dir))

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_available_datasets(self):
        """测试获取可用数据集列表"""
        datasets = self.loader.get_available_datasets()
        self.assertIn(self.test_dataset_name, datasets)
        self.assertIsInstance(datasets, list)

    def test_load_dataset(self):
        """测试加载完整数据集"""
        dataset = self.loader.load_dataset(self.test_dataset_name)

        self.assertIn('clean_data', dataset)
        self.assertIn('dirty_data', dataset)
        self.assertIn('rules', dataset)

        # 验证数据内容
        pd.testing.assert_frame_equal(dataset['clean_data'], self.clean_data)
        pd.testing.assert_frame_equal(dataset['dirty_data'], self.dirty_data)
        self.assertEqual(dataset['rules'], self.test_rules)

    def test_load_clean_data(self):
        """测试只加载clean数据"""
        clean_data = self.loader.load_clean_data(self.test_dataset_name)
        pd.testing.assert_frame_equal(clean_data, self.clean_data)

    def test_load_dirty_data(self):
        """测试只加载dirty数据"""
        dirty_data = self.loader.load_dirty_data(self.test_dataset_name)
        pd.testing.assert_frame_equal(dirty_data, self.dirty_data)

    def test_load_rules(self):
        """测试只加载规则"""
        rules = self.loader.load_rules(self.test_dataset_name)
        self.assertEqual(rules, self.test_rules)

    def test_get_dataset_info(self):
        """测试获取数据集信息"""
        info = self.loader.get_dataset_info(self.test_dataset_name)

        self.assertTrue(info['has_clean_data'])
        self.assertTrue(info['has_dirty_data'])
        self.assertTrue(info['has_rules'])
        self.assertEqual(info['clean_shape'], (5, 3))
        self.assertEqual(info['dirty_shape'], (5, 3))
        self.assertEqual(len(info['columns']), 3)

    def test_load_nonexistent_dataset(self):
        """测试加载不存在的数据集"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_dataset("nonexistent_dataset")


class TestDataManager(unittest.TestCase):
    """测试DataManager类"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时数据集
        self.temp_dir = tempfile.mkdtemp()
        self.test_datasets_dir = Path(self.temp_dir) / "datasets"
        self.test_datasets_dir.mkdir()

        self.test_dataset_name = "test_dataset"
        dataset_dir = self.test_datasets_dir / self.test_dataset_name
        dataset_dir.mkdir()

        # 创建测试数据
        self.clean_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })

        self.dirty_data = pd.DataFrame({
            'col1': [1, 2, 999, 4, 5],
            'col2': ['a', 'b', 'x', 'd', 'e']
        })

        self.clean_data.to_csv(dataset_dir / "clean.csv", index=False)
        self.dirty_data.to_csv(dataset_dir / "dirty.csv", index=False)

        # 创建规则
        self.test_rules = {
            "columns": [
                {
                    "name": "col1",
                    "data_type": "integer",
                    "null_value_rule": "not allowed"
                }
            ]
        }

        with open(dataset_dir / "rule.json", 'w') as f:
            json.dump(self.test_rules, f)

        # 创建DataManager
        loader = DatasetLoader(str(self.test_datasets_dir))
        self.data_manager = DataManager(loader)

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_prepare_data_for_detection(self):
        """测试准备检测数据"""
        prepared_data = self.data_manager.prepare_data_for_detection(self.test_dataset_name)

        self.assertIn('clean_data', prepared_data)
        self.assertIn('dirty_data', prepared_data)
        self.assertIn('error_labels', prepared_data)
        self.assertIn('rules', prepared_data)
        self.assertIn('columns', prepared_data)

        # 验证错误标签
        error_labels = prepared_data['error_labels']
        self.assertEqual(error_labels.loc[2, 'col1'], 1)  # 999是错误值
        self.assertEqual(error_labels.loc[2, 'col2'], 1)  # x是错误值
        self.assertEqual(error_labels.loc[0, 'col1'], 0)  # 1是正确值

    def test_prepare_data_with_sample_size(self):
        """测试带采样的数据准备"""
        prepared_data = self.data_manager.prepare_data_for_detection(
            self.test_dataset_name, sample_size=3
        )

        self.assertEqual(len(prepared_data['clean_data']), 3)
        self.assertEqual(len(prepared_data['dirty_data']), 3)
        self.assertEqual(len(prepared_data['error_labels']), 3)

    def test_generate_error_labels(self):
        """测试错误标签生成"""
        error_labels = self.data_manager._generate_error_labels(
            self.clean_data, self.dirty_data
        )

        # 检查错误标签的正确性
        self.assertEqual(error_labels.loc[2, 'col1'], 1)  # 999 vs 3
        self.assertEqual(error_labels.loc[2, 'col2'], 1)  # x vs c
        self.assertEqual(error_labels.loc[0, 'col1'], 0)  # 1 vs 1
        self.assertEqual(error_labels.loc[0, 'col2'], 0)  # a vs a

    def test_get_column_info(self):
        """测试获取列信息"""
        col_info = self.data_manager.get_column_info(self.test_dataset_name, 'col1')
        self.assertEqual(col_info['name'], 'col1')
        self.assertEqual(col_info['data_type'], 'integer')

    def test_get_data_statistics(self):
        """测试获取数据统计"""
        stats = self.data_manager.get_data_statistics(self.clean_data)

        self.assertEqual(stats['shape'], (5, 2))
        self.assertIn('columns', stats)
        self.assertIn('missing_values', stats)
        self.assertIn('data_types', stats)


if __name__ == '__main__':
    # 运行测试
    print("=== 运行data模块测试 ===")
    unittest.main(verbosity=2)
