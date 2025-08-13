"""
Test data module functionality
Including tests for DatasetLoader
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Add path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from data.loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):
    """Test DatasetLoader class"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directory and test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_datasets_dir = Path(self.temp_dir) / "datasets"
        self.test_datasets_dir.mkdir()

        # Create test dataset
        self.test_dataset_name = "test_dataset"
        dataset_dir = self.test_datasets_dir / self.test_dataset_name
        dataset_dir.mkdir()

        # Create test CSV files
        self.clean_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        }).astype(str)  # Force convert to string type to mimic original implementation

        self.dirty_data = pd.DataFrame({
            'col1': [1, 2, 999, 4, 5],  # 999 is error value
            'col2': ['a', 'b', 'x', 'd', 'e'],  # x is error value
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        }).astype(str)  # Force convert to string type to mimic original implementation

        # Save test data
        self.clean_data.to_csv(dataset_dir / "clean.csv", index=False)
        self.dirty_data.to_csv(dataset_dir / "dirty.csv", index=False)

        # Create rule file - original JSON format
        self.test_rules_json = {
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

        # Expected rule format - format after DatasetLoader processing
        self.test_rules = {
            'col1': {
                'meaning': 'Test integer column',
                'data_type': 'integer',
                'format_rule': 'positive integer',
                'null_value_rule': 'not allowed'
            },
            'col2': {
                'meaning': 'Test string column',
                'data_type': 'string',
                'format_rule': 'single letter',
                'null_value_rule': 'not allowed'
            }
        }

        with open(dataset_dir / "rule.json", 'w') as f:
            json.dump(self.test_rules_json, f)

        self.loader = DatasetLoader(str(self.test_datasets_dir))

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_available_datasets(self):
        """Test getting available dataset list"""
        datasets = self.loader.get_available_datasets()
        self.assertIn(self.test_dataset_name, datasets)
        self.assertIsInstance(datasets, list)

    def test_load_dataset(self):
        """Test loading complete dataset"""
        dataset = self.loader.load_dataset(self.test_dataset_name)

        self.assertIn('clean_data', dataset)
        self.assertIn('dirty_data', dataset)
        self.assertIn('rules', dataset)

        # Verify data content
        pd.testing.assert_frame_equal(dataset['clean_data'], self.clean_data)
        pd.testing.assert_frame_equal(dataset['dirty_data'], self.dirty_data)
        self.assertEqual(dataset['rules'], self.test_rules)

    def test_load_clean_data(self):
        """Test loading clean data only"""
        clean_data = self.loader.load_clean_data(self.test_dataset_name)
        pd.testing.assert_frame_equal(clean_data, self.clean_data)

    def test_load_dirty_data(self):
        """Test loading dirty data only"""
        dirty_data = self.loader.load_dirty_data(self.test_dataset_name)
        pd.testing.assert_frame_equal(dirty_data, self.dirty_data)

    def test_load_rules(self):
        """Test loading rules only"""
        rules = self.loader.load_rules(self.test_dataset_name)
        self.assertEqual(rules, self.test_rules)

    def test_get_dataset_info(self):
        """Test getting dataset information"""
        info = self.loader.get_dataset_info(self.test_dataset_name)

        self.assertTrue(info['has_clean_data'])
        self.assertTrue(info['has_dirty_data'])
        self.assertTrue(info['has_rules'])
        self.assertEqual(info['clean_shape'], (5, 3))
        self.assertEqual(info['dirty_shape'], (5, 3))
        self.assertEqual(len(info['columns']), 3)

    def test_load_nonexistent_dataset(self):
        """Test loading nonexistent dataset"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_dataset("nonexistent_dataset")


if __name__ == '__main__':
    # Run tests
    print("=== Running data module tests ===")
    unittest.main(verbosity=2)
