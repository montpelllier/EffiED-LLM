import json
import random
import re

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import normalized_mutual_info_score

from ed.model import LabelList
from ed.prompt_templates import gen_err_prompt
from llm_wrapper.llm import *


def log_info(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)


def extract_label_list_json(response_content: str) -> str:
    """
    Extracts a JSON string from the response content with a specific pattern.
    """

    json_pattern = r'(\{\s*"labels"\s*:\s*\[[\s\S]*?\]\s*\})'
    match = re.search(json_pattern, response_content)
    if match:
        json_str = match.group(1).strip()
    else:
        print("⚠️ No valid JSON found in response, using raw content")
        json_str = response_content.strip()
        print(json_str)

    return json_str


def chunk_rows_by_length(rows_data: list[dict], max_chars=None, max_rows=None):
    """
    Splits rows of data into chunks based on character length or row count.
    :param rows_data:
    :param max_chars:
    :param max_rows:
    :return:
    """
    if not rows_data:
        return

    if max_chars is None and max_rows is None:
        max_rows = 1

    batch = []
    current_len = 0
    keys = rows_data[0].keys()

    for row in rows_data:
        row_str = '{' + ', '.join(f'"{key}": "{row[key]}"' for key in keys) + '}'
        row_len = len(row_str) + 2

        yeild_flag = False

        if max_chars is not None and (current_len + row_len > max_chars):
            yeild_flag = True

        if max_rows is not None and len(batch) >= max_rows:
            yeild_flag = True

        if yeild_flag and batch:
            yield batch
            batch = []
            current_len = 0

        batch.append(row_str)
        current_len += row_len
    if batch:
        yield batch


def call_llm(model: BaseLLM, prompt=None, messages=None, method="generate", use_thinking=False, model_config=None,
             format_json=None):
    """
    Uses the Ollama API to call a language model with the given prompt or messages.
    """
    # print(prompt)
    if not model_config:
        options = {
            "temperature": 0,
            "seed": 42,
        }
    else:
        options = model_config

    if isinstance(model, OllamaLLM):
        params = {
            'think': use_thinking,
            'format': format_json.model_json_schema(),
            'options': options
        }
        sleep_time = 0
    else:
        params = {
            'temperature': options.get('temperature'),
            'response_format': format_json,
        }
        sleep_time = 10

    if method == "generate":
        response = model.generate(
            prompt=prompt,
            **params,
        )

    elif method == 'chat':  # chat
        if not messages:
            raise ValueError("Messages must be provided for chat method.")
        messages.append({"role": "user", "content": prompt})

        response = model.chat(
            messages=messages,
            think=use_thinking,
            format=LabelList.model_json_schema(),
            options=options
        )
    else:
        raise ValueError("Method must be 'generate' or 'chat'. Please check your input.")

    if sleep_time > 0:
        time.sleep(sleep_time)

    return response


def process_data_chunks(row_list: list[dict], column: str, system_prompt: str, llm: BaseLLM, idx_lst: list[int],
                        max_rows=10, use_thinking=False, fewshot_prompt=None, rule_prompt=None,
                        method=None, logger=None) -> dict[int, bool]:
    """
    Processes data in chunks to detect errors in a specified column using a language model.
    """

    result_dict = {}
    chunk_idx, processed_row_num = 0, 0
    messages = [{"role": "system", "content": system_prompt}]
    prompt = None

    for i in range(0, len(row_list), max_rows):
        chunk_rows = row_list[i:i + max_rows]
        row_ids = [row['row_id'] for row in chunk_rows]
        label_list = {
            "labels": [{"row_id": rid, "is_error": None} for rid in row_ids]
        }

        output_template = f"""
Please fill in the \"is_error\" field (either `true` or `false`) in the following JSON template\n
{json.dumps(label_list, indent=2, ensure_ascii=False)}
"""
        # print(str(label_list))
        chunk = []
        for row in chunk_rows:
            keys = row.keys()
            row_str = '{' + ', '.join(f'"{key}": "{row[key]}"' for key in keys) + '}'
            chunk.append(row_str)

        # for chunk in chunk_rows_by_length(row_list, max_rows=max_rows):
        chunk_json = "[\n\t" + ",\n".join(chunk) + "\n]"
        processed_row_num += len(chunk)
        chunk_idx += 1
        log_info(f"Processing chunk {chunk_idx}: {processed_row_num} / {len(row_list)} rows", logger)

        err_prompt = gen_err_prompt(column, chunk_json, fewshot_prompt, rule_prompt)
        err_prompt += output_template
        # print(err_prompt)

        if method == "generate":
            prompt = system_prompt + "\n\n" + err_prompt
            print(f"prompt length: {len(prompt)}")
        elif method == "chat":
            messages.append({"role": "user", "content": err_prompt})

        if isinstance(llm, OllamaLLM):
            format_json = LabelList
        else:
            format_json = None
        response = call_llm(llm, prompt=prompt, messages=messages, method=method, use_thinking=use_thinking,
                            format_json=format_json)
        json_str = extract_label_list_json(response)
        try:
            parsed = LabelList.model_validate_json(json_str)
        except Exception as e:
            # log_info()
            log_info(f"⚠️ERROR: Failed to parse JSON response: {json_str} with error {e}", logger)
            continue

        if method == "chat":
            messages.append({"role": "assistant", "content": json_str})

        for label in parsed.labels:
            idx = int(label.row_id)
            if idx not in idx_lst:
                log_info(f"⚠️WARNING: Row index {idx} not found in samples for column '{column}'", logger)
                continue
            else:
                result_dict[idx] = label.is_error

        if len(chunk) != len(parsed.labels):
            log_info(
                f"Warning: Chunk {chunk_idx} processed {len(parsed.labels)} labels, but expected {len(chunk)} rows.",
                logger)

    return result_dict


def extract_labels(sample_idx_list: list[int], label_result: dict[int, bool], logger=None) -> tuple[
    list[bool], set[int]]:
    """
    Extracts labels from the result dictionary based on the provided sample indices.
    """

    missing_idx = set()
    label_lst = []

    for idx in sample_idx_list:
        if idx not in label_result:
            log_info(f"⚠️ No label found for index {idx} — default to False", logger)
            label_lst.append(False)
            missing_idx.add(idx)
        else:
            label_lst.append(label_result[idx])

    return label_lst, missing_idx


def generate_data_rows(dataframe: DataFrame, columns: list[str], row_indices: list[int]) -> list[dict]:
    """
    Generates a list of data rows from the specified columns and row indices in the DataFrame.
    """
    rows = []
    for idx in row_indices:
        row_data = {col: dataframe.at[idx, col] for col in columns}
        row_data['row_id'] = idx
        rows.append(row_data)
    return rows


def select_few_shot_examples(dataset, column, num_examples=2, strategy='balanced'):
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
    # nmi_matrix = compute_nmi_matrix(data)
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
