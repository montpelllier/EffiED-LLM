"""
Evaluator
Provides simplified evaluation process and result analysis
"""
from typing import Dict, Any

import pandas

from .metrics import MetricsCalculator


class Evaluator:
    """Error detection result evaluator - simplified version"""

    def __init__(self):
        self.metrics_calculator = MetricsCalculator()

    def evaluate_detection_results(self, y_true: pandas.DataFrame, y_pred: pandas.DataFrame,
                                   dataset_name: str = None, model_name: str = None) -> Dict[str, Any]:
        """
        Evaluate detection results

        Args:
            y_true: True labels
            y_pred: Predicted labels
            dataset_name: Dataset name
            model_name: Model name

        Returns:
            Evaluation result dictionary with overall metrics and per-column results
        """
        # Data validation
        if y_true.shape != y_pred.shape:
            return {'error': f'Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}'}

        if not y_true.columns.equals(y_pred.columns):
            return {'error': 'Column names do not match'}

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred)

        # Build simplified evaluation results
        result = {
            'dataset': dataset_name,
            'model': model_name,
            'overall': {
                'accuracy': metrics['overall']['accuracy'],
                'precision': metrics['overall']['precision'],
                'recall': metrics['overall']['recall'],
                'f1_score': metrics['overall']['f1_score'],
                'total_cells': metrics['overall']['total_cells'],
                'true_error_count': metrics['overall']['true_error_count'],
                'pred_error_count': metrics['overall']['pred_error_count'],
                'true_error_rate': metrics['overall']['true_error_rate'],
                'pred_error_rate': metrics['overall']['pred_error_rate']
            },
            'column_results': metrics['column_wise']
        }

        return result

    def print_results(self, evaluation_result: Dict[str, Any]):
        """
        Print evaluation results

        Args:
            evaluation_result: Result returned by evaluate_detection_results
        """
        if 'error' in evaluation_result:
            print(f"Evaluation Error: {evaluation_result['error']}")
            return

        print("=" * 70)
        print("ERROR DETECTION EVALUATION RESULTS")
        print("=" * 70)

        if evaluation_result.get('dataset'):
            print(f"Dataset: {evaluation_result['dataset']}")
        if evaluation_result.get('model'):
            print(f"Model: {evaluation_result['model']}")
        print()

        # Overall metrics
        overall = evaluation_result['overall']
        print("OVERALL METRICS:")
        print(f"  Accuracy:           {overall['accuracy']:.5f}")
        print(f"  Precision:          {overall['precision']:.5f}")
        print(f"  Recall:             {overall['recall']:.5f}")
        print(f"  F1-Score:           {overall['f1_score']:.5f}")
        print()

        print("ERROR STATISTICS:")
        print(f"  Total Cells:        {overall['total_cells']}")
        print(f"  True Error Count:   {overall['true_error_count']}")
        print(f"  Pred Error Count:   {overall['pred_error_count']}")
        print(f"  True Error Rate:    {overall['true_error_rate']:.5f}")
        print(f"  Pred Error Rate:    {overall['pred_error_rate']:.5f}")
        print()

        # Per-column results
        print("COLUMN-WISE DETAILED RESULTS:")
        print("-" * 70)
        column_results = evaluation_result['column_results']

        for column, metrics in column_results.items():
            print(f"{column}:")
            print(f"  Metrics: Acc={metrics['accuracy']:.5f} | "
                  f"P={metrics['precision']:.5f} | "
                  f"R={metrics['recall']:.5f} | "
                  f"F1={metrics['f1_score']:.5f}")
            print(f"  Errors:  True={metrics['true_error_count']} | "
                  f"Pred={metrics['pred_error_count']} | "
                  f"Total={metrics['total_cells']}")
            print()

        print("=" * 70)
