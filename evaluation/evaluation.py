import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple

from detection.tmp_util import log_info


def evaluate_column_predictions(true_labels, pred_labels):
    """
    Evaluate predictions against true labels and return metrics.

    Parameters:
        true_labels: DataFrame, ground truth error labels
        pred_labels: DataFrame, predicted error labels
    """
    # real_labels = err_labels.iloc[samples][col].tolist()
    result = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, zero_division=0),
        'recall': recall_score(true_labels, pred_labels, zero_division=0),
        'f1': f1_score(true_labels, pred_labels, zero_division=0),
    }
    return result


def evaluate_predictions(true_labels: pd.DataFrame, pred_labels: pd.DataFrame,
                         column_wise: bool = True, overall: bool = True) -> Dict[str, Any]:
    """
    Evaluate prediction results, computing overall and column-wise metrics.

    Parameters:
        true_labels: DataFrame, ground truth error labels
        pred_labels: DataFrame, predicted error labels
        column_wise: bool, whether to compute metrics for each column
        overall: bool, whether to compute overall metrics

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    results = {}

    # Ensure both DataFrames have the same shape
    if true_labels.shape != pred_labels.shape:
        raise ValueError("True labels and predicted labels have mismatched shapes")

    # Compute overall metrics
    if overall:
        # Flatten DataFrames into 1D arrays
        true_flat = true_labels.values.flatten().astype(int)
        pred_flat = pred_labels.values.flatten().astype(int)

        # Compute overall metrics
        results['overall'] = {
            'accuracy': accuracy_score(true_flat, pred_flat),
            'precision': precision_score(true_flat, pred_flat, zero_division=0),
            'recall': recall_score(true_flat, pred_flat, zero_division=0),
            'f1': f1_score(true_flat, pred_flat, zero_division=0),
            'error_count': np.sum(true_flat),
            'predicted_count': np.sum(pred_flat),
            'error_rate': np.sum(true_flat) / len(true_flat) if len(true_flat) > 0 else 0
        }

    # Compute column-wise metrics
    if column_wise:
        column_results = {}
        for col in true_labels.columns:
            true_col = true_labels[col].astype(int).values
            pred_col = pred_labels[col].astype(int).values

            column_results[col] = {
                'accuracy': accuracy_score(true_col, pred_col),
                'precision': precision_score(true_col, pred_col, zero_division=0),
                'recall': recall_score(true_col, pred_col, zero_division=0),
                'f1': f1_score(true_col, pred_col, zero_division=0),
                'error_count': np.sum(true_col),
                'predicted_count': np.sum(pred_col),
                'error_rate': np.sum(true_col) / len(true_col) if len(true_col) > 0 else 0
            }

        results['columns'] = column_results

    return results


def print_evaluation_results(results: Dict[str, Any], print_columns: bool = True, logger=None) -> None:
    """
    Print evaluation results in a formatted table.

    Parameters:
        results: dict, return value from evaluate_predictions
        print_columns: bool, whether to print column-wise results
    """
    # Print overall results
    if 'overall' in results:
        overall = results['overall']
        overall_info = (
            f"Overall Evaluation Results:\n"
            f"Accuracy:  {overall['accuracy']:.5f}\n"
            f"Precision: {overall['precision']:.5f}\n"
            f"Recall:    {overall['recall']:.5f}\n"
            f"F1 Score:  {overall['f1']:.5f}\n"
            f"True Error Count: {overall['error_count']}\n"
            f"Predicted Count:  {overall['predicted_count']}\n"
            f"Error Rate: {overall['error_rate']*100:.3f} %\n"
        )
        log_info(overall_info, logger)

    # Print column-wise results
    if 'columns' in results and print_columns:
        log_info("\nColumn-wise Evaluation Results:", logger)
        columns = results['columns']

        # Create results table
        headers = ["Column", "Accuracy", "Precision", "Recall", "F1", "Err. Cnt", "Predicted", "Err. Rate"]
        rows = []

        for col, metrics in columns.items():
            row = [
                col,
                f"{metrics['accuracy']:.5f}",
                f"{metrics['precision']:.5f}",
                f"{metrics['recall']:.5f}",
                f"{metrics['f1']:.5f}",
                metrics['error_count'],
                metrics['predicted_count'],
                f"{metrics['error_rate']*100:.3f} %"
            ]
            rows.append(row)

        # Calculate column widths
        col_widths = [max(len(h), max([len(str(row[i])) for row in rows])) for i, h in enumerate(headers)]

        # Print header
        header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print("header_str")
        log_info(header_str, logger)
        log_info("-" * len(header_str), logger)

        # Print data rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            log_info(row_str, logger)


def summarize_results_by_column(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame summarizing metrics for each column.

    Parameters:
        true_labels: DataFrame, ground truth error labels
        pred_labels: DataFrame, predicted error labels

    Returns:
        DataFrame: Summary metrics for each column
    """
    results = evaluate_predictions(true_labels, pred_labels)

    if 'columns' not in results:
        return pd.DataFrame()

    summary = []
    for col, metrics in results['columns'].items():
        summary.append({
            'column': col,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'error_count': metrics['error_count'],
            'predicted_count': metrics['predicted_count']
        })

    return pd.DataFrame(summary)


def evaluate_model(true_labels: pd.DataFrame, pred_labels: pd.DataFrame, logger=None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Comprehensive model evaluation - computes metrics and returns both detailed results and summary.

    Parameters:
        true_labels: DataFrame, ground truth error labels
        pred_labels: DataFrame, predicted error labels

    Returns:
        tuple: (detailed results dictionary, summary DataFrame)
    """
    # Compute all metrics
    results = evaluate_predictions(true_labels, pred_labels)

    # Print formatted results
    print_evaluation_results(results, logger=logger)

    # Generate summary DataFrame
    summary_df = summarize_results_by_column(true_labels, pred_labels)

    return results, summary_df