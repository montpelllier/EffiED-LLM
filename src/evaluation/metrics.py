"""
Metrics Calculator
Calculate basic evaluation metrics for error detection
"""
from typing import Dict, Any

import pandas
import sklearn


class MetricsCalculator:
    """Evaluation metrics calculator - simplified version"""

    def __init__(self):
        pass

    @staticmethod
    def calculate_basic_metrics(y_true: pandas.Series, y_pred: pandas.Series) -> Dict[str, float]:
        """
        Calculate basic classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
            'precision': sklearn.metrics.precision_score(y_true, y_pred, zero_division=0),
            'recall': sklearn.metrics.recall_score(y_true, y_pred, zero_division=0),
            'f1_score': sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)
        }

        return metrics

    def calculate_column_wise_metrics(self, y_true_df: pandas.DataFrame,
                                      y_pred_df: pandas.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate column-wise metrics

        Args:
            y_true_df: True labels DataFrame
            y_pred_df: Predicted labels DataFrame

        Returns:
            Dictionary of metrics for each column
        """
        column_metrics = {}

        for column in y_true_df.columns:
            if column in y_pred_df.columns:
                y_true_col = y_true_df[column]
                y_pred_col = y_pred_df[column]

                # Basic metrics
                basic_metrics = self.calculate_basic_metrics(y_true_col, y_pred_col)

                # Add error counts
                true_error_count = int(y_true_col.sum())
                pred_error_count = int(y_pred_col.sum())
                total_cells = len(y_true_col)

                # Merge metrics
                column_metrics[column] = {
                    **basic_metrics,
                    'true_error_count': true_error_count,
                    'pred_error_count': pred_error_count,
                    'total_cells': total_cells
                }

        return column_metrics

    def calculate_overall_metrics(self, y_true_df: pandas.DataFrame,
                                  y_pred_df: pandas.DataFrame) -> Dict[str, float]:
        """
        Calculate overall metrics (treating all cells as a whole)
        """
        # Flatten all labels
        y_true_flat = y_true_df.values.flatten()
        y_pred_flat = y_pred_df.values.flatten()

        # Remove NaN values
        mask = ~(pandas.isna(y_true_flat) | pandas.isna(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0:
            raise ValueError("No valid data for evaluation")

        # Calculate basic metrics
        basic_metrics = self.calculate_basic_metrics(
            pandas.Series(y_true_clean), pandas.Series(y_pred_clean)
        )

        # Calculate error counts and rates
        total_cells = len(y_true_clean)
        true_error_count = int(sum(y_true_clean))
        pred_error_count = int(sum(y_pred_clean))
        true_error_rate = true_error_count / total_cells if total_cells > 0 else 0.0
        pred_error_rate = pred_error_count / total_cells if total_cells > 0 else 0.0

        return {
            **basic_metrics,
            'total_cells': total_cells,
            'true_error_count': true_error_count,
            'pred_error_count': pred_error_count,
            'true_error_rate': true_error_rate,
            'pred_error_rate': pred_error_rate
        }

    def calculate_metrics(self, y_true: pandas.DataFrame, y_pred: pandas.DataFrame) -> Dict[str, Any]:
        """
        Calculate the complete set of evaluation metrics
        """
        results = {'overall': self.calculate_overall_metrics(y_true, y_pred),
                   'column_wise': self.calculate_column_wise_metrics(y_true, y_pred)}

        return results
