import random
import re

import ollama
from pandas import DataFrame

from detection.model import LabelList
from detection.prompt_templates import gen_err_prompt


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


def call_llm(model_name: str, prompt=None, messages=None, method="generate", use_thinking=False, model_config=None):
    """
    Uses the Ollama API to call a language model with the given prompt or messages.
    """
    if not model_name:
        options = {
            "temperature": 0,
            "seed": 42,
        }
    else:
        options = model_config

    if method == "generate":
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            format=LabelList.model_json_schema(),
            think=use_thinking,
            options=options
        ).response
        return response

    elif method == 'chat':  # chat
        if not messages:
            raise ValueError("Messages must be provided for chat method.")
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=model_name,
            messages=messages,
            format=LabelList.model_json_schema(),
            think=use_thinking,
            options=options
        ).message
        return response.content
    else:
        raise ValueError("Method must be 'generate' or 'chat'. Please check your input.")


def process_data_chunks(rows: list[dict], col: str, system_prompt: str, model_name: str, idx_lst: list[int],
                        max_rows=10, use_thinking=False, fewshot_prompt=None, rule_prompt=None,
                        method=None, logger=None) -> dict[int, bool]:
    """
    Processes data in chunks to detect errors in a specified column using a language model.
    """

    result_dict = {}
    chunk_idx, processed_row_num = 0, 0
    messages = [{"role": "system", "content": system_prompt}]
    prompt = None

    for chunk in chunk_rows_by_length(rows, max_rows=max_rows):
        chunk_json = "[\n\t" + ",\n".join(chunk) + "\n]"
        processed_row_num += len(chunk)
        chunk_idx += 1
        log_info(f"Processing chunk {chunk_idx}: {processed_row_num} / {len(rows)} rows", logger)

        err_prompt = gen_err_prompt(col, chunk_json, fewshot_prompt, rule_prompt)

        if method == "generate":
            prompt = system_prompt + "\n\n" + err_prompt
        elif method == "chat":
            messages.append({"role": "user", "content": err_prompt})

        response = call_llm(model_name, prompt=prompt, messages=messages, method=method, use_thinking=use_thinking)
        json_str = extract_label_list_json(response)
        parsed = LabelList.model_validate_json(json_str)

        if method == "chat":
            messages.append({"role": "assistant", "content": json_str})

        for label in parsed.labels:
            idx = int(label.row_id)
            if idx not in idx_lst:
                log_info(f"⚠️WARNING: Row index {idx} not found in samples for column '{col}'", logger)
                continue
            else:
                result_dict[idx] = label.is_error

        if len(chunk) != len(parsed.labels):
            log_info(f"Warning: Chunk {chunk_idx} processed {len(parsed.labels)} labels, but expected {len(chunk)} rows.", logger)

    return result_dict


def extract_labels(sample_idx_list: list[int], label_result: dict[int, bool], logger=None) -> tuple[list[bool], set[int]]:
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