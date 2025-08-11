"""
测试detection模块中的feature_extractor功能
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from detection.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """测试FeatureExtractor类"""

    def setUp(self):
        """设置测试环境"""
        self.extractor = FeatureExtractor()

        # 创建测试数据
        self.numeric_series = pd.Series([1, 2, 3, 4, 100])  # 包含异常值
        self.string_series = pd.Series(['abc', 'def', 'ghi', 'x', 'abcdef'])  # 不同长度
        self.mixed_series = pd.Series(['123', '456', 'abc', '789'])
        self.email_series = pd.Series(['test@email.com', 'user@domain.org', 'invalid', 'another@test.com'])
        self.date_series = pd.Series(['2023-01-01', '2023-02-15', '2023-03-20', 'invalid-date'])
        self.null_series = pd.Series([1, 2, None, 4, None])

    def test_extract_basic_stats(self):
        """测试基本统计特征提取"""
        stats = self.extractor._extract_basic_stats(self.numeric_series)

        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['non_null_count'], 5)
        self.assertEqual(stats['null_count'], 0)
        self.assertEqual(stats['unique_count'], 5)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)

        # 检查统计值的正确性
        self.assertEqual(stats['mean'], 22.0)  # (1+2+3+4+100)/5
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 100)

    def test_extract_data_type_features(self):
        """测试数据类型特征提取"""
        # 测试数值型数据
        numeric_features = self.extractor._extract_data_type_features(self.numeric_series)
        self.assertTrue(numeric_features['is_numeric'])
        self.assertFalse(numeric_features['is_string'])

        # 测试字符串数据
        string_features = self.extractor._extract_data_type_features(self.string_series)
        self.assertFalse(string_features['is_numeric'])
        self.assertTrue(string_features['is_string'])

        # 测试推断类型
        mixed_features = self.extractor._extract_data_type_features(self.mixed_series)
        self.assertIn('inferred_types', mixed_features)
        self.assertIn('dominant_type', mixed_features)

    def test_extract_pattern_features(self):
        """测试模式特征提取"""
        pattern_features = self.extractor._extract_pattern_features(self.string_series)

        self.assertIn('pattern_diversity', pattern_features)
        self.assertIn('common_patterns', pattern_features)
        self.assertIn('has_consistent_format', pattern_features)

        # 检查模式多样性
        self.assertGreater(pattern_features['pattern_diversity'], 0)
        self.assertIsInstance(pattern_features['common_patterns'], list)

    def test_extract_outlier_features(self):
        """测试异常值特征提取"""
        outlier_features = self.extractor._extract_outlier_features(self.numeric_series)

        self.assertIn('potential_outliers_count', outlier_features)
        self.assertIn('length_outliers', outlier_features)

        # 100应该被识别为异常值
        self.assertGreater(outlier_features['potential_outliers_count'], 0)

    def test_extract_completeness_features(self):
        """测试完整性特征提取"""
        # 测试有null值的数据
        completeness_features = self.extractor._extract_completeness_features(self.null_series)

        self.assertIn('completeness_ratio', completeness_features)
        self.assertEqual(completeness_features['completeness_ratio'], 0.6)  # 3/5

        # 测试无null值的数据
        complete_features = self.extractor._extract_completeness_features(self.numeric_series)
        self.assertEqual(complete_features['completeness_ratio'], 1.0)

    def test_infer_value_type(self):
        """测试值类型推断"""
        self.assertEqual(self.extractor._infer_value_type('123'), 'integer')
        self.assertEqual(self.extractor._infer_value_type('123.45'), 'float')
        self.assertEqual(self.extractor._infer_value_type('abc'), 'string')
        self.assertEqual(self.extractor._infer_value_type('test@email.com'), 'email')
        self.assertEqual(self.extractor._infer_value_type('2023-01-01'), 'date')
        self.assertEqual(self.extractor._infer_value_type(None), 'null')

    def test_extract_char_pattern(self):
        """测试字符模式提取"""
        # 测试不同类型的模式
        self.assertEqual(self.extractor._extract_char_pattern('123'), 'DDD')
        self.assertEqual(self.extractor._extract_char_pattern('abc'), 'LLL')
        self.assertEqual(self.extractor._extract_char_pattern('ABC'), 'UUU')
        self.assertEqual(self.extractor._extract_char_pattern('a1b2'), 'LDLD')
        self.assertEqual(self.extractor._extract_char_pattern('a b'), 'LSL')
        self.assertEqual(self.extractor._extract_char_pattern('a-b'), 'LPL')

    def test_looks_like_date(self):
        """测试日期识别"""
        self.assertTrue(self.extractor._looks_like_date('2023-01-01'))
        self.assertTrue(self.extractor._looks_like_date('01/15/2023'))
        self.assertTrue(self.extractor._looks_like_date('01-15-2023'))
        self.assertFalse(self.extractor._looks_like_date('not-a-date'))
        self.assertFalse(self.extractor._looks_like_date('123456'))

    def test_extract_column_features_comprehensive(self):
        """测试完整的列特征提取"""
        # 测试数值列
        numeric_features = self.extractor.extract_column_features(self.numeric_series)

        # 验证包含所有特征类别
        expected_keys = [
            'count', 'non_null_count', 'null_count', 'unique_count',
            'is_numeric', 'is_string', 'pattern_diversity',
            'completeness_ratio', 'potential_outliers_count'
        ]

        for key in expected_keys:
            self.assertIn(key, numeric_features, f"Missing feature: {key}")

        # 测试字符串列
        string_features = self.extractor.extract_column_features(self.string_series)
        self.assertTrue(string_features['is_string'])
        self.assertFalse(string_features['is_numeric'])

    def test_email_detection(self):
        """测试邮箱检测"""
        email_features = self.extractor.extract_column_features(self.email_series)

        # 检查推断类型中是否包含email类型
        inferred_types = email_features.get('inferred_types', {})
        self.assertIn('email', inferred_types)

        # 验证email类型被识别
        self.assertGreater(inferred_types.get('email', 0), 0)


if __name__ == '__main__':
    print("=== 运行detection.feature_extractor测试 ===")
    unittest.main(verbosity=2)
