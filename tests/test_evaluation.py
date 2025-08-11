"""
测试evaluation模块的功能
包括MetricsCalculator、Evaluator和ReportGenerator的测试
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# 添加路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from evaluation import MetricsCalculator, Evaluator, ReportGenerator


class TestMetricsCalculator(unittest.TestCase):
    """测试MetricsCalculator类"""

    def setUp(self):
        """设置测试环境"""
        self.calculator = MetricsCalculator()

        # 创建测试数据
        self.y_true_perfect = pd.Series([0, 1, 0, 1, 0])
        self.y_pred_perfect = pd.Series([0, 1, 0, 1, 0])  # 完美预测

        self.y_true_mixed = pd.Series([0, 1, 0, 1, 0, 1])
        self.y_pred_mixed = pd.Series([0, 0, 1, 1, 0, 0])  # 混合结果

        # DataFrame测试数据
        self.df_true = pd.DataFrame({
            'col1': [0, 1, 0, 1],
            'col2': [1, 0, 1, 0],
            'col3': [0, 0, 1, 1]
        })

        self.df_pred = pd.DataFrame({
            'col1': [0, 1, 1, 1],  # 1个FP
            'col2': [1, 0, 0, 0],  # 1个FN
            'col3': [0, 0, 1, 0]   # 1个FN
        })

    def test_calculate_basic_metrics_perfect(self):
        """测试完美预测的基本指标计算"""
        metrics = self.calculator.calculate_basic_metrics(
            self.y_true_perfect, self.y_pred_perfect
        )

        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)

    def test_calculate_basic_metrics_mixed(self):
        """测试混合结果的基本指标计算"""
        metrics = self.calculator.calculate_basic_metrics(
            self.y_true_mixed, self.y_pred_mixed
        )

        self.assertAlmostEqual(metrics['accuracy'], 0.5, places=2)  # 3/6 correct
        self.assertAlmostEqual(metrics['precision'], 0.5, places=2)  # 1/2 TP/(TP+FP)
        self.assertAlmostEqual(metrics['recall'], 0.333, places=2)   # 1/3 TP/(TP+FN)

        # F1应该是precision和recall的调和平均
        expected_f1 = 2 * (0.5 * 0.333) / (0.5 + 0.333)
        self.assertAlmostEqual(metrics['f1_score'], expected_f1, places=2)

    def test_calculate_confusion_matrix_metrics(self):
        """测试混淆矩阵指标计算"""
        cm_metrics = self.calculator.calculate_confusion_matrix_metrics(
            self.y_true_mixed, self.y_pred_mixed
        )

        self.assertIn('confusion_matrix', cm_metrics)
        self.assertIn('true_negatives', cm_metrics)
        self.assertIn('false_positives', cm_metrics)
        self.assertIn('false_negatives', cm_metrics)
        self.assertIn('true_positives', cm_metrics)

        # 验证混淆矩阵的值
        tn, fp, fn, tp = (cm_metrics['true_negatives'],
                         cm_metrics['false_positives'],
                         cm_metrics['false_negatives'],
                         cm_metrics['true_positives'])

        self.assertEqual(tn + fp + fn + tp, len(self.y_true_mixed))

    def test_calculate_column_wise_metrics(self):
        """测试列级别指标计算"""
        column_metrics = self.calculator.calculate_column_wise_metrics(
            self.df_true, self.df_pred
        )

        # 验证返回所有列的指标
        for col in self.df_true.columns:
            self.assertIn(col, column_metrics)
            self.assertIn('accuracy', column_metrics[col])
            self.assertIn('precision', column_metrics[col])
            self.assertIn('recall', column_metrics[col])
            self.assertIn('f1_score', column_metrics[col])

    def test_calculate_overall_metrics(self):
        """测试整体指标计算"""
        overall_metrics = self.calculator.calculate_overall_metrics(
            self.df_true, self.df_pred
        )

        self.assertIn('accuracy', overall_metrics)
        self.assertIn('precision', overall_metrics)
        self.assertIn('recall', overall_metrics)
        self.assertIn('f1_score', overall_metrics)
        self.assertIn('total_cells', overall_metrics)
        self.assertIn('error_cells', overall_metrics)
        self.assertIn('detected_errors', overall_metrics)

        # 验证总单元格数
        expected_total = self.df_true.size
        self.assertEqual(overall_metrics['total_cells'], expected_total)

    def test_calculate_summary_metrics(self):
        """测试汇总指标计算"""
        column_metrics = self.calculator.calculate_column_wise_metrics(
            self.df_true, self.df_pred
        )

        summary_metrics = self.calculator.calculate_summary_metrics(column_metrics)

        # 验证包含平均值、标准差、最小值、最大值
        self.assertIn('avg_accuracy', summary_metrics)
        self.assertIn('std_accuracy', summary_metrics)
        self.assertIn('min_f1_score', summary_metrics)
        self.assertIn('max_precision', summary_metrics)

    def test_calculate_error_distribution(self):
        """测试错误分布计算"""
        error_dist = self.calculator.calculate_error_distribution(
            self.df_true, self.df_pred
        )

        self.assertIn('true_error_distribution', error_dist)
        self.assertIn('predicted_error_distribution', error_dist)

        true_dist = error_dist['true_error_distribution']
        pred_dist = error_dist['predicted_error_distribution']

        # 验证错误分布结构
        self.assertIn('total_errors', true_dist)
        self.assertIn('errors_per_column', true_dist)
        self.assertIn('errors_per_row_stats', true_dist)

        self.assertIn('total_errors', pred_dist)
        self.assertIn('errors_per_column', pred_dist)
        self.assertIn('errors_per_row_stats', pred_dist)

    def test_calculate_metrics_comprehensive(self):
        """测试完整指标计算"""
        results = self.calculator.calculate_metrics(self.df_true, self.df_pred)

        # 验证包含所有主要部分
        self.assertIn('overall', results)
        self.assertIn('column_wise', results)
        self.assertIn('summary', results)
        self.assertIn('error_distribution', results)

    def test_edge_cases(self):
        """测试边界情况"""
        # 全零预测
        y_true_zeros = pd.Series([0, 0, 0, 0])
        y_pred_zeros = pd.Series([0, 0, 0, 0])

        metrics_zeros = self.calculator.calculate_basic_metrics(
            y_true_zeros, y_pred_zeros, zero_division=0
        )

        self.assertEqual(metrics_zeros['accuracy'], 1.0)
        self.assertEqual(metrics_zeros['precision'], 0)  # zero_division处理

        # 全一预测
        y_true_ones = pd.Series([1, 1, 1, 1])
        y_pred_ones = pd.Series([1, 1, 1, 1])

        metrics_ones = self.calculator.calculate_basic_metrics(
            y_true_ones, y_pred_ones
        )

        self.assertEqual(metrics_ones['accuracy'], 1.0)
        self.assertEqual(metrics_ones['precision'], 1.0)


class TestEvaluator(unittest.TestCase):
    """测试Evaluator类"""

    def setUp(self):
        """设置测试环境"""
        self.evaluator = Evaluator()

        # 测试数据
        self.y_true = pd.DataFrame({
            'col1': [0, 1, 0, 1],
            'col2': [1, 0, 1, 0]
        })

        self.y_pred = pd.DataFrame({
            'col1': [0, 1, 1, 1],
            'col2': [1, 0, 0, 0]
        })

    def test_evaluate_detection_results(self):
        """测试完整的评估流程"""
        result = self.evaluator.evaluate_detection_results(
            self.y_true, self.y_pred,
            dataset_name="test_dataset",
            model_name="test_model",
            detection_config={"mode": "test"}
        )

        # 验证结构
        self.assertIn('metadata', result)
        self.assertIn('metrics', result)
        self.assertIn('analysis', result)

        # 验证元数据
        metadata = result['metadata']
        self.assertEqual(metadata['dataset_name'], "test_dataset")
        self.assertEqual(metadata['model_name'], "test_model")
        self.assertIn('evaluation_timestamp', metadata)

    def test_validate_input_data(self):
        """测试输入数据验证"""
        # 正常情况
        validation = self.evaluator._validate_input_data(self.y_true, self.y_pred)
        self.assertTrue(validation['valid'])

        # 形状不匹配
        y_pred_wrong_shape = pd.DataFrame({'col1': [0, 1]})
        validation = self.evaluator._validate_input_data(self.y_true, y_pred_wrong_shape)
        self.assertFalse(validation['valid'])
        self.assertIn('Shape mismatch', validation['message'])

        # 非法值
        y_pred_invalid = pd.DataFrame({
            'col1': [0, 1, 2, 1],  # 2是非法值
            'col2': [1, 0, 0, 0]
        })
        validation = self.evaluator._validate_input_data(self.y_true, y_pred_invalid)
        self.assertFalse(validation['valid'])

    def test_classify_performance(self):
        """测试性能分类"""
        excellent_metrics = {'f1_score': 0.95}
        self.assertEqual(self.evaluator._classify_performance(excellent_metrics), 'excellent')

        good_metrics = {'f1_score': 0.85}
        self.assertEqual(self.evaluator._classify_performance(good_metrics), 'good')

        poor_metrics = {'f1_score': 0.3}
        self.assertEqual(self.evaluator._classify_performance(poor_metrics), 'poor')

    def test_analyze_error_types(self):
        """测试错误类型分析"""
        analysis = self.evaluator._analyze_error_types(self.y_true, self.y_pred)

        self.assertIn('true_positives', analysis)
        self.assertIn('false_positives', analysis)
        self.assertIn('false_negatives', analysis)
        self.assertIn('detection_rate', analysis)
        self.assertIn('false_alarm_rate', analysis)

        # 验证值的合理性
        self.assertGreaterEqual(analysis['detection_rate'], 0)
        self.assertLessEqual(analysis['detection_rate'], 1)

    def test_generate_recommendations(self):
        """测试建议生成"""
        # 低精确率情况
        low_precision_metrics = {
            'overall': {'precision': 0.5, 'recall': 0.8, 'f1_score': 0.6}
        }

        recommendations = self.evaluator._generate_recommendations(low_precision_metrics)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # 检查是否包含精确率相关建议
        rec_text = ' '.join(recommendations)
        self.assertIn('精确率', rec_text)

    def test_evaluation_history(self):
        """测试评估历史记录"""
        initial_count = len(self.evaluator.evaluation_history)

        # 进行一次评估
        self.evaluator.evaluate_detection_results(self.y_true, self.y_pred)

        # 验证历史记录增加
        self.assertEqual(len(self.evaluator.evaluation_history), initial_count + 1)

    def test_get_evaluation_summary(self):
        """测试获取评估摘要"""
        # 先进行一次评估
        self.evaluator.evaluate_detection_results(self.y_true, self.y_pred, model_name="test_model")

        # 获取摘要
        summary = self.evaluator.get_evaluation_summary()

        self.assertIn('model', summary)
        self.assertIn('accuracy', summary)
        self.assertIn('precision', summary)
        self.assertIn('recall', summary)
        self.assertIn('f1_score', summary)
        self.assertIn('performance_level', summary)


class TestReportGenerator(unittest.TestCase):
    """测试ReportGenerator类"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.report_generator = ReportGenerator(self.temp_dir)

        # 创建测试评估结果
        self.evaluation_result = {
            'metadata': {
                'dataset_name': 'test_dataset',
                'model_name': 'test_model',
                'evaluation_timestamp': '2023-01-01T12:00:00',
                'data_shape': (100, 5)
            },
            'metrics': {
                'overall': {
                    'accuracy': 0.85,
                    'precision': 0.80,
                    'recall': 0.75,
                    'f1_score': 0.77,
                    'true_positives': 30,
                    'false_positives': 10,
                    'false_negatives': 15,
                    'true_negatives': 45
                },
                'column_wise': {
                    'col1': {'accuracy': 0.9, 'precision': 0.85, 'recall': 0.8, 'f1_score': 0.82},
                    'col2': {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.7, 'f1_score': 0.72}
                }
            },
            'analysis': {
                'performance_level': 'good',
                'recommendations': ['建议1', '建议2']
            }
        }

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_generate_json_report(self):
        """测试JSON报告生成"""
        report_path = self.report_generator._generate_json_report(
            self.evaluation_result, "test_report"
        )

        # 验证文件存在
        self.assertTrue(Path(report_path).exists())
        self.assertTrue(report_path.endswith('.json'))

        # 验证内容正确
        with open(report_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        self.assertEqual(loaded_data['metadata']['dataset_name'], 'test_dataset')
        self.assertEqual(loaded_data['metrics']['overall']['accuracy'], 0.85)

    def test_generate_html_report(self):
        """测试HTML报告生成"""
        report_path = self.report_generator._generate_html_report(
            self.evaluation_result, "test_report"
        )

        # 验证文件存在
        self.assertTrue(Path(report_path).exists())
        self.assertTrue(report_path.endswith('.html'))

        # 验证HTML内容
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('test_dataset', html_content)
        self.assertIn('test_model', html_content)
        self.assertIn('0.85', html_content)  # accuracy

    def test_generate_markdown_report(self):
        """测试Markdown报告生成"""
        report_path = self.report_generator._generate_markdown_report(
            self.evaluation_result, "test_report"
        )

        # 验证文件存在
        self.assertTrue(Path(report_path).exists())
        self.assertTrue(report_path.endswith('.md'))

        # 验证Markdown内容
        with open(report_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        self.assertIn('# 错误检测评估报告', md_content)
        self.assertIn('test_dataset', md_content)
        self.assertIn('| 准确率', md_content)
        self.assertIn('0.85', md_content)

    def test_generate_detailed_report(self):
        """测试详细报告生成"""
        # 测试不同格式
        formats = ['json', 'html', 'markdown']

        for format_type in formats:
            with self.subTest(format_type=format_type):
                report_path = self.report_generator.generate_detailed_report(
                    self.evaluation_result, format_type
                )

                self.assertTrue(Path(report_path).exists())
                self.assertIn(format_type.replace('markdown', 'md'), report_path)

    def test_unsupported_format(self):
        """测试不支持的格式"""
        with self.assertRaises(ValueError):
            self.report_generator.generate_detailed_report(
                self.evaluation_result, 'unsupported_format'
            )


if __name__ == '__main__':
    print("=== 运行evaluation模块测试 ===")
    unittest.main(verbosity=2)
