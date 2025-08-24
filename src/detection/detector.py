import json
import time
from typing import Dict, List

import tqdm
from pandas import DataFrame
from pydantic import BaseModel

import llm
from data import Dataset
from .utils import compute_nmi_matrix, get_top_nmi_relations, generate_data_rows, select_few_shot_examples, \
    extract_label_list_json


class ErrorLabel(BaseModel):
    row_id: int
    is_error: bool


class LabelList(BaseModel):
    labels: list[ErrorLabel]


class ErrorDetector:

    def __init__(self, dataset: Dataset, representatives_dataframe: DataFrame, model: llm.BaseLLM,
                 strategies: List[str], fewshot: int, batch_size: int):

        self.dataset = dataset
        self.llm = model
        self.representatives = representatives_dataframe

        self.fewshot = fewshot
        self.batch_size = batch_size
        self._apply_prompt_strategies(strategies)

    def _apply_prompt_strategies(self, strategies):
        dirty_data = self.dataset.dirty_data

        self.column_map = {}
        self.prompt_config = {}
        for strategy_name in strategies:
            if strategy_name == 'ZS-I':
                self.prompt_config['include_rule'] = False
            elif strategy_name == 'ZS-R':
                self.prompt_config['include_rule'] = True
            elif strategy_name == 'V':
                for column in dirty_data.columns:
                    self.column_map[column] = [column]
            elif strategy_name == 'A':
                nmi_matrix = compute_nmi_matrix(dirty_data)
                related_col_dict = get_top_nmi_relations(nmi_matrix, max_attr=5, min_attr=1)
                for column in dirty_data.columns:
                    self.column_map[column] = [column] + related_col_dict[column]

        self.prompt_config['include_fewshot'] = self.fewshot > 0

    def detect_errors(self) -> Dict[str, Dict[int, bool]]:
        result = {}
        for column in tqdm.tqdm(self.dataset.dirty_data.columns):
            result[column] = self._detect_column_errors(column)

        return result

    def _detect_column_errors(self, column_name: str) -> Dict[int, bool]:
        column_result = {}
        ditry_data = self.dataset.dirty_data

        column_lst = self.column_map[column_name]
        column_repre_idx = self.representatives[column_name].astype(int).tolist()
        data_rows = generate_data_rows(ditry_data, column_lst, column_repre_idx)

        for i in range(0, len(data_rows), self.batch_size):
            batch_data = data_rows[i:i + self.batch_size]
            row_ids = [row['row_id'] for row in batch_data]
            label_list = {
                "labels": [{"row_id": rid, "is_error": None} for rid in row_ids]
            }
            output_json = json.dumps(label_list, indent=2, ensure_ascii=False)
            batch_json = json.dumps(batch_data, indent=2, ensure_ascii=False)
            fewshot_examples = None
            if self.fewshot > 0:
                fewshot_examples = select_few_shot_examples(self.dataset, column_name, self.fewshot,
                                                            strategy='balanced')

            prompt = llm.prompt_manager.generate_error_detection_prompt(column_name=column_name,
                                                                        data_json=batch_json,
                                                                        label_json=output_json,
                                                                        fewshot_examples=fewshot_examples,
                                                                        rule_content=self.dataset.rules[column_name],
                                                                        **self.prompt_config)

            delay = 0
            if isinstance(self.llm, llm.OllamaLLM):
                llm_config = {
                    'think': False,
                    'format': LabelList.model_json_schema(),
                    'options': {
                        'temperature': 0,
                        'seed': 42
                    }
                }
            elif isinstance(self.llm, llm.OpenAILLM):
                llm_config = {
                    'temperature': 0,
                }
                delay = 10
            else:
                llm_config = {}

            llm_response = self.llm.generate(prompt, **llm_config)
            if delay > 0:
                time.sleep(delay)

            json_str = extract_label_list_json(llm_response)
            try:
                parsed = LabelList.model_validate_json(json_str)
            except Exception as e:
                print(f"⚠️ERROR: Failed to parse JSON response: {json_str} with error {e}")
                continue

            for label in parsed.labels:
                idx = int(label.row_id)
                if idx not in column_repre_idx:
                    print(f"⚠️WARNING: Row index {idx} not found in samples for column '{column_name}'")
                    continue
                else:
                    column_result[idx] = label.is_error

        return column_result
