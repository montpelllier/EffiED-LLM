import random
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def compute_nmi_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    data_copy = dataframe.copy()
    nmi_matrix = pd.DataFrame(index=dataframe.columns, columns=dataframe.columns, dtype=float)

    for col1 in dataframe.columns:
        for col2 in dataframe.columns:
            valid_idx = data_copy[[col1, col2]].dropna().index
            if len(valid_idx) > 0:
                x = data_copy.loc[valid_idx, col1]
                y = data_copy.loc[valid_idx, col2]
                nmi = normalized_mutual_info_score(x, y)
                nmi_matrix.loc[col1, col2] = nmi
            else:
                nmi_matrix.loc[col1, col2] = np.nan  # 无有效数据

    return nmi_matrix


def get_top_nmi_relations(nmi_matrix, threshold=0.9, max_attr=3, min_attr=1):
    assert min_attr <= max_attr, "min_attr should be less than or equal to max_attr"

    result = {}
    for col in nmi_matrix.columns:
        scores = nmi_matrix.loc[col].drop(index=col)  # 排除自己
        sorted_scores = scores.sort_values(ascending=False)
        high_nmi = sorted_scores[sorted_scores > threshold]

        if len(high_nmi) >= max_attr > 0:
            selected = high_nmi.head(max_attr)
        elif len(high_nmi) >= min_attr > 0:
            selected = high_nmi
        elif min_attr > len(high_nmi):
            selected = sorted_scores.head(min_attr)
        else:
            selected = pd.Series(dtype=float)

        result[col] = list(selected.index)

    return result


def generate_data_rows(dataframe: pd.DataFrame, columns: list[str], row_indices: list[int]) -> list[dict]:
    """
    Generates a list of data rows from the specified columns and row indices in the DataFrame.
    """
    rows = []
    for idx in row_indices:
        row_data = {col: dataframe.at[idx, col] for col in columns}
        row_data['row_id'] = idx
        rows.append(row_data)
    return rows


def extract_label_list_json(response_content: str) -> str:
    """
    Extracts a JSON string from the response content with a specific pattern.
    """

    json_pattern = r'(\{\s*"labels"\s*:\s*\[[\s\S]*?\]\s*\})'
    match = re.search(json_pattern, response_content)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = response_content.strip()
        print(f"⚠️ No valid JSON found in response, using raw content:\n{json_str}")

    return json_str


def select_few_shot_examples(dataset, column, num_examples=2, strategy='balanced') -> List[Dict]:
    """
    Select few-shot examples for the given column

    Args:
        dataset: Dataset object containing dirty_data, clean_data, and error_labels
        column: Column name to select examples for
        num_examples: Number of examples to select
        strategy: Selection strategy ('random', 'diverse', 'balanced')

    Returns:
        List of example dictionaries
    """
    dirty_data = dataset.dirty_data
    clean_data = dataset.clean_data
    error_labels = dataset.error_labels

    examples = []

    if num_examples <= 0 or num_examples > len(error_labels):
        return examples

    if strategy == 'balanced':
        # Try to get balanced examples (both error and non-error)
        error_indices = error_labels[error_labels[column] == True].index.tolist()
        clean_indices = error_labels[error_labels[column] == False].index.tolist()

        # Get roughly half error and half clean examples
        num_error = min(len(error_indices), num_examples)
        num_clean = num_examples - num_error

        selected_error = random.sample(error_indices, num_error)
        selected_clean = random.sample(clean_indices, num_clean)
        selected_indices = selected_error + selected_clean

    elif strategy == 'diverse':
        # Select diverse examples based on value uniqueness
        unique_values = dirty_data[column].unique()
        selected_indices = []

        for value in random.sample(list(unique_values), min(len(unique_values), num_examples)):
            value_indices = dirty_data[dirty_data[column] == value].index.tolist()
            if value_indices:
                selected_indices.append(random.choice(value_indices))

        # Fill remaining with random if needed
        if len(selected_indices) < num_examples:
            remaining_indices = [i for i in error_labels.index if i not in selected_indices]
            additional = random.sample(remaining_indices,
                                       min(len(remaining_indices), num_examples - len(selected_indices)))
            selected_indices.extend(additional)

    else:  # random
        selected_indices = random.sample(list(error_labels.index),
                                         min(len(error_labels.index), num_examples))

    # Create example dictionaries
    for idx in selected_indices:
        example = {}
        for col in dirty_data.columns:
            example[col] = dirty_data.loc[idx, col]

        example['clean_value'] = clean_data.loc[idx, column]
        examples.append(example)

    return examples
