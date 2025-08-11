"""
评估器
提供完整的评估流程和结果分析
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .metrics import MetricsCalculator


class Evaluator:
    """错误检测结果评估器"""

    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.evaluation_history = []

    def evaluate_detection_results(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                                 dataset_name: str = None, model_name: str = None,
                                 detection_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        全面评估检测结果

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            dataset_name: 数据集名称
            model_name: 模型名称
            detection_config: 检测配置

        Returns:
            评估结果字典
        """
        # 数据验证
        validation_result = self._validate_input_data(y_true, y_pred)
        if not validation_result['valid']:
            return {'error': validation_result['message']}

        # 计算指标
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred)

        # 构建评估结果
        evaluation_result = {
            'metadata': {
                'dataset_name': dataset_name,
                'model_name': model_name,
                'detection_config': detection_config,
                'data_shape': y_true.shape,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'metrics': metrics,
            'analysis': self._analyze_results(y_true, y_pred, metrics)
        }

        # 保存到历史记录
        self.evaluation_history.append(evaluation_result)

        return evaluation_result

    def _validate_input_data(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, Any]:
        """验证输入数据的有效性"""

        # 检查数据类型
        if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
            return {'valid': False, 'message': 'Input data must be pandas DataFrames'}

        # 检查形状是否匹配
        if y_true.shape != y_pred.shape:
            return {'valid': False, 'message': f'Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}'}

        # 检查列名是否匹配
        if not y_true.columns.equals(y_pred.columns):
            return {'valid': False, 'message': 'Column names do not match'}

        # 检查索引是否匹配
        if not y_true.index.equals(y_pred.index):
            return {'valid': False, 'message': 'Index does not match'}

        # 检查值是否在有效范围内
        valid_values = {0, 1}
        true_values = set(y_true.values.flatten())
        pred_values = set(y_pred.values.flatten())

        # 移除NaN值进行检查
        true_values.discard(np.nan)
        pred_values.discard(np.nan)

        if not true_values.issubset(valid_values):
            return {'valid': False, 'message': f'y_true contains invalid values: {true_values - valid_values}'}

        if not pred_values.issubset(valid_values):
            return {'valid': False, 'message': f'y_pred contains invalid values: {pred_values - valid_values}'}

        return {'valid': True, 'message': 'Data validation passed'}

    def _analyze_results(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                        metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析评估结果，提供深入洞察"""
        analysis = {}

        # 性能分析
        overall_metrics = metrics.get('overall', {})
        analysis['performance_level'] = self._classify_performance(overall_metrics)

        # 错误类型分析
        analysis['error_analysis'] = self._analyze_error_types(y_true, y_pred)

        # 列级别分析
        column_metrics = metrics.get('column_wise', {})
        analysis['column_analysis'] = self._analyze_column_performance(column_metrics)

        # 建议和改进方向
        analysis['recommendations'] = self._generate_recommendations(metrics)

        return analysis

    def _classify_performance(self, metrics: Dict[str, float]) -> str:
        """根据指标对性能进行分类"""
        f1_score = metrics.get('f1_score', 0)

        if f1_score >= 0.9:
            return 'excellent'
        elif f1_score >= 0.8:
            return 'good'
        elif f1_score >= 0.7:
            return 'fair'
        elif f1_score >= 0.5:
            return 'poor'
        else:
            return 'very_poor'

    def _analyze_error_types(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, Any]:
        """分析错误类型"""
        # 计算不同类型的错误
        tp = ((y_true == 1) & (y_pred == 1)).sum().sum()  # 正确检测到的错误
        fp = ((y_true == 0) & (y_pred == 1)).sum().sum()  # 误报
        fn = ((y_true == 1) & (y_pred == 0)).sum().sum()  # 漏报
        tn = ((y_true == 0) & (y_pred == 0)).sum().sum()  # 正确识别的正常数据

        total_errors = (y_true == 1).sum().sum()
        total_normal = (y_true == 0).sum().sum()

        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'total_actual_errors': int(total_errors),
            'total_actual_normal': int(total_normal),
            'detection_rate': float(tp / total_errors) if total_errors > 0 else 0.0,
            'false_alarm_rate': float(fp / total_normal) if total_normal > 0 else 0.0
        }

    def _analyze_column_performance(self, column_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """分析列级别的性能"""
        if not column_metrics:
            return {}

        # 找出表现最好和最差的列
        f1_scores = {col: metrics.get('f1_score', 0) for col, metrics in column_metrics.items()}

        best_column = max(f1_scores, key=f1_scores.get)
        worst_column = min(f1_scores, key=f1_scores.get)

        # 计算列之间的性能差异
        f1_values = list(f1_scores.values())
        performance_variance = np.var(f1_values)

        return {
            'best_performing_column': {
                'name': best_column,
                'f1_score': f1_scores[best_column]
            },
            'worst_performing_column': {
                'name': worst_column,
                'f1_score': f1_scores[worst_column]
            },
            'performance_variance': float(performance_variance),
            'consistent_performance': performance_variance < 0.01  # 性能是否一致
        }

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """基于评估结果生成改进建议"""
        recommendations = []

        overall_metrics = metrics.get('overall', {})
        precision = overall_metrics.get('precision', 0)
        recall = overall_metrics.get('recall', 0)
        f1_score = overall_metrics.get('f1_score', 0)

        # 基于精确率和召回率的建议
        if precision < 0.7:
            recommendations.append("精确率较低，建议调整检测阈值或改进提示词以减少误报")

        if recall < 0.7:
            recommendations.append("召回率较低，建议增加更多示例或改进错误模式识别")

        if f1_score < 0.6:
            recommendations.append("总体性能较低，建议尝试不同的检测策略或模型")

        # 基于列性能差异的建议
        column_analysis = metrics.get('analysis', {}).get('column_analysis', {})
        if not column_analysis.get('consistent_performance', True):
            recommendations.append("列之间性能差异较大，建议为不同列使用专门的检测策略")

        if not recommendations:
            recommendations.append("检测性能良好，可以考虑进一步优化以提高效率")

        return recommendations

    def compare_evaluations(self, evaluation_ids: List[int]) -> Dict[str, Any]:
        """比较多个评估结果"""
        if len(evaluation_ids) < 2:
            return {'error': 'Need at least 2 evaluations to compare'}

        evaluations = []
        for eval_id in evaluation_ids:
            if 0 <= eval_id < len(self.evaluation_history):
                evaluations.append(self.evaluation_history[eval_id])
            else:
                return {'error': f'Invalid evaluation ID: {eval_id}'}

        comparison = {
            'evaluations_count': len(evaluations),
            'metrics_comparison': self._compare_metrics(evaluations),
            'performance_ranking': self._rank_evaluations(evaluations)
        }

        return comparison

    def _compare_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较多个评估的指标"""
        comparison = {}

        # 提取各评估的主要指标
        for i, evaluation in enumerate(evaluations):
            overall_metrics = evaluation.get('metrics', {}).get('overall', {})
            model_name = evaluation.get('metadata', {}).get('model_name', f'Model_{i}')

            comparison[model_name] = {
                'accuracy': overall_metrics.get('accuracy', 0),
                'precision': overall_metrics.get('precision', 0),
                'recall': overall_metrics.get('recall', 0),
                'f1_score': overall_metrics.get('f1_score', 0)
            }

        return comparison

    def _rank_evaluations(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按F1分数对评估结果进行排名"""
        ranked = []

        for i, evaluation in enumerate(evaluations):
            overall_metrics = evaluation.get('metrics', {}).get('overall', {})
            model_name = evaluation.get('metadata', {}).get('model_name', f'Model_{i}')
            f1_score = overall_metrics.get('f1_score', 0)

            ranked.append({
                'model_name': model_name,
                'f1_score': f1_score,
                'evaluation_id': i
            })

        # 按F1分数降序排列
        ranked.sort(key=lambda x: x['f1_score'], reverse=True)

        return ranked

    def get_evaluation_summary(self, evaluation_id: int = -1) -> Dict[str, Any]:
        """获取评估结果摘要"""
        if not self.evaluation_history:
            return {'error': 'No evaluations available'}

        if evaluation_id < 0:
            evaluation_id = len(self.evaluation_history) - 1

        if evaluation_id >= len(self.evaluation_history):
            return {'error': 'Invalid evaluation ID'}

        evaluation = self.evaluation_history[evaluation_id]
        overall_metrics = evaluation.get('metrics', {}).get('overall', {})

        summary = {
            'dataset': evaluation.get('metadata', {}).get('dataset_name'),
            'model': evaluation.get('metadata', {}).get('model_name'),
            'accuracy': overall_metrics.get('accuracy', 0),
            'precision': overall_metrics.get('precision', 0),
            'recall': overall_metrics.get('recall', 0),
            'f1_score': overall_metrics.get('f1_score', 0),
            'performance_level': evaluation.get('analysis', {}).get('performance_level'),
            'recommendations': evaluation.get('analysis', {}).get('recommendations', [])
        }

        return summary
