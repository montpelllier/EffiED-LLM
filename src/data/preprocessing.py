"""
Data Preprocessing Module
Responsible for cleaning and preprocessing datasets before error detection.
Handles removal of empty columns, constant columns, and dataset-specific processing.
"""
from typing import Dict, Any, Tuple

import pandas as pd


class DataPreprocessor:
    """
    Data preprocessor for cleaning datasets before error detection.

    This class handles various preprocessing tasks including:
    - Removing completely empty columns
    - Removing constant columns (same value across all rows)
    - Applying dataset-specific column removal rules
    - Statistical analysis of column quality
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize the DataPreprocessor.

        Args:
            enable_logging: Whether to enable detailed logging of preprocessing steps
        """
        self.enable_logging = enable_logging
        self.preprocessing_stats = {}

    def preprocess_dataset(self, dirty_data: pd.DataFrame, clean_data: pd.DataFrame = None,
                           dataset_name: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess a complete dataset by applying various cleaning rules.

        Args:
            dirty_data: Dirty data with errors (required)
            clean_data: Clean reference data (optional, if None will use dirty_data as reference)
            dataset_name: Name of the dataset for logging and specific rules

        Returns:
            Tuple of (processed_dirty_data, processed_clean_data, error_labels, preprocessing_stats)

        Note:
            If clean_data is None, dirty_data will be used for both clean and dirty processing,
            and error labels cannot be generated accurately.
        """
        if dataset_name is None:
            dataset_name = "unknown"

        # If no clean data provided, use dirty data as reference for preprocessing decisions
        if clean_data is None:
            clean_data = dirty_data.copy()
            clean_data_provided = False
            if self.enable_logging:
                print(f"Preprocessing dataset '{dataset_name}' (dirty data only)...")
        else:
            clean_data_provided = True
            if self.enable_logging:
                print(f"Preprocessing dataset '{dataset_name}' (clean + dirty data)...")

        # Initialize stats
        stats = {
            'dataset_name': dataset_name,
            'clean_data_provided': clean_data_provided,
            'original_shape': dirty_data.shape,
            'removed_columns': [],
            'removal_reasons': {},
            'final_shape': None
        }

        # Ensure column names match
        clean_data.columns = dirty_data.columns

        # Apply preprocessing steps in order
        clean_processed, dirty_processed = self._remove_empty_columns(clean_data.copy(), dirty_data.copy(), stats)
        clean_processed, dirty_processed = self._remove_constant_columns(clean_processed, dirty_processed, stats)
        # Update final stats
        stats['final_shape'] = dirty_processed.shape
        stats['columns_retained'] = list(dirty_processed.columns)
        stats['removal_summary'] = len(stats['removed_columns'])

        error_labels = self._cal_error_labels(clean_processed, dirty_processed)
        stats['error_ratio'] = error_labels.mean().mean() if clean_data_provided else None
        # Log results if enabled
        if self.enable_logging:
            self._log_preprocessing_results(stats)

        # Store stats for later access
        self.preprocessing_stats[dataset_name] = stats

        return dirty_processed, clean_processed, error_labels, stats

    def _remove_empty_columns(self, clean_data: pd.DataFrame, dirty_data: pd.DataFrame,
                              stats: Dict[str, Any], max_empty_ratio: float = 0.95) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Remove columns that are completely empty or mostly empty.

        Args:
            clean_data: Clean data DataFrame
            dirty_data: Dirty data DataFrame
            stats: Statistics dictionary to update
            max_empty_ratio: Maximum allowed ratio of empty values (default: 0.95)
        """
        columns_to_drop = []

        for col in dirty_data.columns:
            # Check dirty data - count NaN and empty strings
            dirty_empty_count = dirty_data[col].isna().sum()
            dirty_blank_count = (dirty_data[col].astype(str).str.strip() == '').sum()
            dirty_empty_ratio = (dirty_empty_count + dirty_blank_count) / len(dirty_data)

            # Check clean data - count NaN and empty strings
            clean_empty_count = clean_data[col].isna().sum()
            clean_blank_count = (clean_data[col].astype(str).str.strip() == '').sum()
            clean_empty_ratio = (clean_empty_count + clean_blank_count) / len(clean_data)

            # Remove if mostly empty in either dataset
            if dirty_empty_ratio >= max_empty_ratio or clean_empty_ratio >= max_empty_ratio:
                columns_to_drop.append(col)
                stats['removal_reasons'][
                    col] = f"Empty ratio: dirty={dirty_empty_ratio:.3f}, clean={clean_empty_ratio:.3f}"

        if columns_to_drop and self.enable_logging:
            print(f"Removing empty columns: {columns_to_drop}")

        if columns_to_drop:
            clean_data = clean_data.drop(columns=columns_to_drop)
            dirty_data = dirty_data.drop(columns=columns_to_drop)
            stats['removed_columns'].extend(columns_to_drop)

        return clean_data, dirty_data

    def _remove_constant_columns(self, clean_data: pd.DataFrame, dirty_data: pd.DataFrame,
                                 stats: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove columns where all values are the same (constant columns).

        This handles the case mentioned for hospital dataset where dirty and clean
        columns have the same constant value across all rows.
        """
        columns_to_drop = []

        for col in dirty_data.columns:
            # Check if column has only one unique value (ignoring NaN and empty strings)
            dirty_non_empty = dirty_data[col].dropna()
            dirty_non_empty = dirty_non_empty[dirty_non_empty.astype(str).str.strip() != '']
            dirty_unique = dirty_non_empty.nunique()

            clean_non_empty = clean_data[col].dropna()
            clean_non_empty = clean_non_empty[clean_non_empty.astype(str).str.strip() != '']
            clean_unique = clean_non_empty.nunique()

            # Remove if constant in both datasets (or effectively empty)
            if dirty_unique <= 1 and clean_unique <= 1:
                columns_to_drop.append(col)

                # Get the constant values for logging
                dirty_val = dirty_non_empty.iloc[0] if len(dirty_non_empty) > 0 else "Empty/NaN"
                clean_val = clean_non_empty.iloc[0] if len(clean_non_empty) > 0 else "Empty/NaN"
                stats['removal_reasons'][col] = f"Constant column: dirty='{dirty_val}', clean='{clean_val}'"

        if columns_to_drop and self.enable_logging:
            print(f"Removing constant columns: {columns_to_drop}")

        if columns_to_drop:
            clean_data = clean_data.drop(columns=columns_to_drop)
            dirty_data = dirty_data.drop(columns=columns_to_drop)
            stats['removed_columns'].extend(columns_to_drop)

        return clean_data, dirty_data

    @staticmethod
    def _cal_error_labels(clean_data: pd.DataFrame, dirty_data: pd.DataFrame, ) -> pd.DataFrame:
        return dirty_data.ne(clean_data)  # True if dirty != clean, False if they are equal

    @staticmethod
    def _log_preprocessing_results(stats: Dict[str, Any]) -> None:
        """Log preprocessing results to console."""
        print(f"\nPreprocessing Results for '{stats['dataset_name']}':")
        print(f"  Original shape: {stats['original_shape']}")
        print(f"  Final shape: {stats['final_shape']}")
        print(f"  Removed {stats['removal_summary']} columns: {stats['removed_columns']}")
        print(f"  Clean data provided: {stats['clean_data_provided']}")
        print(f"  Error ratio: {stats.get('error_ratio', 'N/A') * 100:.3f}%" if stats.get(
            'error_ratio') is not None else "  Error ratio: N/A")

        if stats['removal_reasons']:
            print("  Removal reasons:")
            for col, reason in stats['removal_reasons'].items():
                print(f"    {col}: {reason}")

    def get_preprocessing_stats(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Get preprocessing statistics for a dataset.

        Args:
            dataset_name: Name of dataset, if None returns all stats

        Returns:
            Dictionary containing preprocessing statistics
        """
        if dataset_name is None:
            return self.preprocessing_stats
        return self.preprocessing_stats.get(dataset_name, {})


# Convenience functions
def preprocess_dataset(dirty_data: pd.DataFrame, clean_data: pd.DataFrame = None,
                       dataset_name: str = None, enable_logging: bool = True) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to preprocess a dataset.

    Args:
        dirty_data: Dirty data with errors (required)
        clean_data: Clean reference data (optional)
        dataset_name: Name of the dataset
        enable_logging: Whether to enable logging

    Returns:
        Tuple of (processed_dirty_data, processed_clean_data, error_labels, stats)
    """
    preprocessor = DataPreprocessor(enable_logging=enable_logging)
    return preprocessor.preprocess_dataset(dirty_data, clean_data, dataset_name)
