import logging
import sys
from pathlib import Path

import pandas


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
        rule_path = dataset_path / 'rule.json'

        self.clean_data = pandas.read_csv(clean_path, dtype=str)
        self.dirty_data = pandas.read_csv(dirty_path, dtype=str)
        self.clean_data.columns = self.dirty_data.columns  # Ensure columns match

        if rule_path.exists():
            rules = pandas.read_json(rule_path)
            columns_df = pandas.json_normalize(rules['columns'])
            self.rules = columns_df.set_index('name').to_dict(orient='index')

        if dataset == 'hospital':
            columns_to_drop = ['address_2', 'address_3']
            self.dirty_data = self.dirty_data.drop(columns=columns_to_drop)
            self.clean_data = self.clean_data.drop(columns=columns_to_drop)
        self.error_labels = self.clean_data != self.dirty_data

    # def __repr__(self):
    #     return f"Dataset(data={self.data})"


def get_logger(config):
    model_name = config.get('model_name')
    dataset_name = config.get('dataset_name')
    use_thinking = config.get('use_thinking')
    few_shot = config.get('few_shot')
    use_attr = config.get('use_attr')
    method = config.get('method')
    max_rows = config.get('max_rows')

    safe_model_name = model_name.replace(':', '_')  # 将冒号替换为下划线
    think = "think" if use_thinking else "no-think"
    context = "attr" if use_attr else "value"
    time = pandas.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_filename = f'logs/{safe_model_name}/{dataset_name}_{think}_{few_shot}shot ({context},{method},batch={max_rows}) {time}.log'

    current_dir = Path(__file__).parent
    log_filename = current_dir / log_filename
    print(log_filename)
    log_dir = log_filename.parent
    # log_dir = Path(log_filename).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
