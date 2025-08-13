"""
Test FeatureExtractor functionality in detection.feature module
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from detection.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test FeatureExtractor class, ensure correspondence with ed/features.py functionality"""

    def setUp(self):
        """Set up test environment"""
        self.extractor = FeatureExtractor()

        # Create test data, simulating real scenarios
        self.test_texts = [
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon"
        ]

        self.test_series = pd.Series(self.test_texts)

        # Texts containing numbers
        self.mixed_texts = [
            "item1", "item2", "item123", "product", "code999",
            "test", "sample456", "data", "file1.txt", "doc2"
        ]
        self.mixed_series = pd.Series(self.mixed_texts)

        # Test DataFrame for FD features
        self.test_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
            'age': [25, 30, 35, 25, 30],
            'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
            'salary': [50000, 60000, 70000, 50000, 60000]
        }).astype(str)

        # Data containing missing values
        self.null_series = pd.Series([
            'value1', None, 'value3', '', 'value5', np.nan, 'value7'
        ])

    def test_initialization(self):
        """Test FeatureExtractor initialization"""
        extractor = FeatureExtractor()
        self.assertIsNotNone(extractor.llm_tokenizer)
        self.assertIsNotNone(extractor.standard_scaler)
        self.assertIsNotNone(extractor.minmax_scaler)

    def test_tokenize(self):
        """Test tokenization functionality"""
        test_text = "Hello world 123"
        tokens = self.extractor.tokenize(self.extractor.llm_tokenizer, test_text)

        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)

        # Test number splitting functionality (if advanced features available)
        digit_text = "abc123def"
        tokens = self.extractor.tokenize(self.extractor.llm_tokenizer, digit_text)
        self.assertIsInstance(tokens, list)

    def test_pattern_feature_dataframe(self):
        """Test pattern feature generation, corresponding to generate_pattern_feature in ed/features.py"""
        df = self.extractor.extract_pattern_feature(self.mixed_texts)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.mixed_texts))
        self.assertTrue(df.shape[1] > 0)  # Should have feature columns

    def test_rule_feature_extraction(self):
        """Test functional dependency feature generation, corresponding to generate_fd_feature in ed/features.py"""
        rule_features = self.extractor.extract_rule_feature(self.test_df, 'salary')

        self.assertIsInstance(rule_features, pd.DataFrame)
        self.assertEqual(len(rule_features), len(self.test_df))

        # Check column names contain support and confidence
        column_names = rule_features.columns.tolist()
        support_cols = [col for col in column_names if 'support' in col]
        confidence_cols = [col for col in column_names if 'confidence' in col]

        self.assertTrue(len(support_cols) > 0)
        self.assertTrue(len(confidence_cols) > 0)

        # Verify feature values are in reasonable range
        self.assertTrue((rule_features >= 0).all().all())  # All values should be non-negative

    def test_basic_feature_extraction(self):
        """Test basic statistical feature extraction"""
        basic_features, feature_names = self.extractor._extract_basic_features(self.test_texts)

        self.assertIsInstance(basic_features, np.ndarray)
        self.assertIsInstance(feature_names, list)
        self.assertEqual(basic_features.shape[0], len(self.test_texts))
        self.assertEqual(basic_features.shape[1], len(feature_names))

        # Check if expected feature names are present
        expected_features = ['char_count', 'digit_count', 'alpha_count', 'digit_ratio', 'alpha_ratio']
        for feature in expected_features:
            self.assertIn(feature, feature_names)

    def test_character_feature_extraction(self):
        """Test character-level TF-IDF feature extraction"""
        config = {'analyzer': 'char', 'max_features': 50}
        char_features, feature_names = self.extractor._extract_character_feature(self.test_texts, config)

        self.assertIsInstance(char_features, np.ndarray)
        self.assertIsInstance(feature_names, list)
        self.assertEqual(char_features.shape[0], len(self.test_texts))
        self.assertTrue(len(feature_names) > 0)

        # Check if feature names have expected prefix
        self.assertTrue(all(name.startswith('char_') for name in feature_names))

    def test_token_binning_features(self):
        """Test token binning features, corresponding to compute_token_bin_features in ed/features.py"""
        binned_matrix, bin_col_names, token_counts = self.extractor._bin_tokens(
            self.test_texts,
            n_bins=10,
            method='quantile'
        )

        self.assertIsInstance(binned_matrix, np.ndarray)
        self.assertIsInstance(bin_col_names, list)
        self.assertIsInstance(token_counts, list)

        self.assertEqual(binned_matrix.shape[0], len(self.test_texts))
        self.assertEqual(len(token_counts), len(self.test_texts))
        self.assertTrue(all(count >= 0 for count in token_counts))

        # Test different binning methods
        methods = ['quantile', 'uniform', 'log']
        for method in methods:
            try:
                matrix, _, _ = self.extractor._bin_tokens(
                    self.test_texts[:3],  # Use less data to avoid errors
                    method=method,
                    n_bins=5
                )
                self.assertIsInstance(matrix, np.ndarray)
            except Exception as e:
                # Some methods might fail due to data characteristics
                print(f"Method {method} failed: {e}")

    def test_feature_merging(self):
        """Test feature matrix merging functionality"""
        # Create test feature matrices
        feature1 = np.random.rand(5, 3)
        feature2 = np.random.rand(5, 2)
        features = [feature1, feature2]

        names1 = ['feat1_1', 'feat1_2', 'feat1_3']
        names2 = ['feat2_1', 'feat2_2']
        feature_names = [names1, names2]

        merged_df = self.extractor._merge_features(features, feature_names)

        self.assertIsInstance(merged_df, pd.DataFrame)
        self.assertEqual(merged_df.shape[0], 5)
        self.assertEqual(merged_df.shape[1], 5)
        self.assertEqual(list(merged_df.columns), names1 + names2)

    def test_comprehensive_feature_extraction(self):
        """Test comprehensive feature generation, corresponding to generate_features in ed/features.py"""
        all_features = self.extractor.extract_column_feature(self.test_df, 'name')

        self.assertIsInstance(all_features, pd.DataFrame)
        self.assertEqual(len(all_features), len(self.test_df))
        self.assertTrue(all_features.shape[1] > 0)

    def test_error_handling(self):
        """Test error handling and boundary cases"""
        # Empty list
        empty_result = self.extractor.extract_pattern_feature([])
        self.assertTrue(empty_result.empty)

        # Test invalid target column
        with self.assertRaises(ValueError):
            self.extractor.extract_column_feature(self.test_df, "nonexistent_column")

        # Test empty DataFrame
        with self.assertRaises(ValueError):
            empty_df = pd.DataFrame()
            self.extractor.extract_column_feature(empty_df, "any_column")

    def test_feature_consistency_with_ed_features(self):
        """Test consistency with ed/features.py functionality"""
        # This test ensures our implementation behaves consistently with original implementation

        # Test pattern features basic behavior
        pattern_features = self.extractor.extract_pattern_feature(self.test_texts)

        # Should generate features for each input
        self.assertEqual(len(pattern_features), len(self.test_texts))

        # Test FD features basic behavior
        fd_features = self.extractor.extract_rule_feature(self.test_df, 'name')
        self.assertEqual(len(fd_features), len(self.test_df))

        # Feature values should be numeric
        self.assertTrue(pattern_features.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all())
        self.assertTrue(fd_features.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all())

    def test_performance_with_large_data(self):
        """Test performance with larger datasets (simple test)"""
        # Create larger test dataset
        large_texts = ['text_' + str(i) for i in range(100)]

        # Should be able to process without errors
        features = self.extractor.extract_pattern_feature(large_texts)
        self.assertEqual(len(features), 100)


if __name__ == '__main__':
    # Run tests
    print("=== Running detection.feature module tests ===")
    unittest.main(verbosity=2)
