"""
指标计算器
计算各种评估指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self):
        pass

    def calculate_basic_metrics(self, y_true: pd.Series, y_pred: pd.Series,
                              zero_division: int = 0) -> Dict[str, float]:
        """
        计算基本分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            zero_division: 除零时的默认值

        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=zero_division),
            'recall': recall_score(y_true, y_pred, zero_division=zero_division),
            'f1_score': f1_score(y_true, y_pred, zero_division=zero_division)
        }

        return metrics

    def calculate_confusion_matrix_metrics(self, y_true: pd.Series,
                                         y_pred: pd.Series) -> Dict[str, Any]:
        """
        基于混淆矩阵计算详细指标
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # 处理只有一个类别的情况
            tn = fp = fn = tp = 0
            if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
                if y_true.iloc[0] == y_pred.iloc[0] == 0:
                    tn = len(y_true)
                elif y_true.iloc[0] == y_pred.iloc[0] == 1:
                    tp = len(y_true)

        # 计算派生指标
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr
        }

    def calculate_column_wise_metrics(self, y_true_df: pd.DataFrame,
                                    y_pred_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        计算列级别的指标

        Args:
            y_true_df: 真实标签DataFrame
            y_pred_df: 预测标签DataFrame

        Returns:
            每列的指标字典
        """
        column_metrics = {}

        for column in y_true_df.columns:
            if column in y_pred_df.columns:
                y_true_col = y_true_df[column]
                y_pred_col = y_pred_df[column]

                # 基本指标
                basic_metrics = self.calculate_basic_metrics(y_true_col, y_pred_col)

                # 混淆矩阵指标
                cm_metrics = self.calculate_confusion_matrix_metrics(y_true_col, y_pred_col)

                # 合并指标
                column_metrics[column] = {
                    **basic_metrics,
                    'true_positives': cm_metrics['true_positives'],
                    'false_positives': cm_metrics['false_positives'],
                    'false_negatives': cm_metrics['false_negatives'],
                    'true_negatives': cm_metrics['true_negatives'],
                    'specificity': cm_metrics['specificity'],
                    'sensitivity': cm_metrics['sensitivity']
                }

        return column_metrics

    def calculate_overall_metrics(self, y_true_df: pd.DataFrame,
                                y_pred_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算整体指标（将所有单元格视为一个整体）
        """
        # 展平所有标签
        y_true_flat = y_true_df.values.flatten()
        y_pred_flat = y_pred_df.values.flatten()

        # 移除NaN值
        mask = ~(pd.isna(y_true_flat) | pd.isna(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0:
            return {'error': 'No valid data for evaluation'}

        # 计算基本指标
        basic_metrics = self.calculate_basic_metrics(
            pd.Series(y_true_clean), pd.Series(y_pred_clean)
        )

        # 计算混淆矩阵指标
        cm_metrics = self.calculate_confusion_matrix_metrics(
            pd.Series(y_true_clean), pd.Series(y_pred_clean)
        )

        return {
            **basic_metrics,
            'total_cells': len(y_true_clean),
            'error_cells': sum(y_true_clean),
            'detected_errors': sum(y_pred_clean),
            'true_positives': cm_metrics['true_positives'],
            'false_positives': cm_metrics['false_positives'],
            'false_negatives': cm_metrics['false_negatives'],
            'true_negatives': cm_metrics['true_negatives']
        }

    def calculate_summary_metrics(self, column_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        计算汇总指标（各列指标的平均值）
        """
        if not column_metrics:
            return {}

        # 收集所有指标名称
        metric_names = set()
        for col_metrics in column_metrics.values():
            metric_names.update(col_metrics.keys())

        summary = {}
        for metric_name in metric_names:
            values = []
            for col_metrics in column_metrics.values():
                if metric_name in col_metrics and not pd.isna(col_metrics[metric_name]):
                    values.append(col_metrics[metric_name])

            if values:
                summary[f'avg_{metric_name}'] = np.mean(values)
                summary[f'std_{metric_name}'] = np.std(values)
                summary[f'min_{metric_name}'] = np.min(values)
                summary[f'max_{metric_name}'] = np.max(values)

        return summary

    def calculate_error_distribution(self, y_true_df: pd.DataFrame,
                                   y_pred_df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算错误分布统计
        """
        stats = {}

        # 真实错误分布
        true_errors_per_column = y_true_df.sum()
        true_errors_per_row = y_true_df.sum(axis=1)

        # 预测错误分布
        pred_errors_per_column = y_pred_df.sum()
        pred_errors_per_row = y_pred_df.sum(axis=1)

        stats['true_error_distribution'] = {
            'total_errors': y_true_df.sum().sum(),
            'errors_per_column': true_errors_per_column.to_dict(),
            'errors_per_row_stats': {
                'mean': true_errors_per_row.mean(),
                'std': true_errors_per_row.std(),
                'max': true_errors_per_row.max(),
                'min': true_errors_per_row.min()
            }
        }

        stats['predicted_error_distribution'] = {
            'total_errors': y_pred_df.sum().sum(),
            'errors_per_column': pred_errors_per_column.to_dict(),
            'errors_per_row_stats': {
                'mean': pred_errors_per_row.mean(),
                'std': pred_errors_per_row.std(),
                'max': pred_errors_per_row.max(),
                'min': pred_errors_per_row.min()
            }
        }

        return stats

    def calculate_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, Any]:
        """
        计算完整的评估指标集合
        """
        results = {}

        # 整体指标
        results['overall'] = self.calculate_overall_metrics(y_true, y_pred)

        # 列级指标
        results['column_wise'] = self.calculate_column_wise_metrics(y_true, y_pred)

        # 汇总指标
        results['summary'] = self.calculate_summary_metrics(results['column_wise'])

        # 错误分布
        results['error_distribution'] = self.calculate_error_distribution(y_true, y_pred)

        return results
