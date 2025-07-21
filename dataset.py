import pandas
from pathlib import Path

valid_dataset_lst = ['flights', 'movies_1', 'beers', 'hospital', 'rayyan']


class Dataset:
    def __init__(self, dataset):
        if dataset not in valid_dataset_lst:
            raise ValueError(f"Invalid dataset name: {dataset}. Valid options are: {valid_dataset_lst}")
        # 获取dataset.py文件所在目录
        current_dir = Path(__file__).parent
        dataset_path = current_dir / 'datasets' / dataset
        clean_path = dataset_path / 'clean.csv'
        dirty_path = dataset_path / 'dirty.csv'

        self.clean_data = pandas.read_csv(clean_path, dtype=str)
        self.dirty_data = pandas.read_csv(dirty_path, dtype=str)
        self.clean_data.columns = self.dirty_data.columns  # Ensure columns match

        if dataset == 'hospital':
            columns_to_drop = ['address_2', 'address_3']
            self.dirty_data = self.dirty_data.drop(columns=columns_to_drop)
            self.clean_data = self.clean_data.drop(columns=columns_to_drop)
        self.error_labels = self.clean_data != self.dirty_data


    # def __repr__(self):
    #     return f"Dataset(data={self.data})"
