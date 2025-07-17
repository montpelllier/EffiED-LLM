import re

import ollama

from detection.model import LabelList
from detection.prompt_templates import gen_err_prompt


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


def chunk_rows_by_length(rows_data: dict, max_chars=None, max_rows=None):
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


def call_llm(model_name: str, prompt=None, messages=None, method="generate", use_thinking=False):
    """
    Uses the Ollama API to call a language model with the given prompt or messages.
    """
    if method == "generate":
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            format=LabelList.model_json_schema(),
            think=use_thinking,
            options={"temperature": 0},
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
            options={"temperature": 0},
        ).message
        return response.content
    else:
        raise ValueError("Method must be 'generate' or 'chat'. Please check your input.")


def process_data_chunks(rows: dict, col: str, system_prompt: str, model_name: str, idx_lst: list[int],
                        use_thinking=False, max_rows=10,
                        method="generate") -> dict[int, bool]:
    """
    Processes data in chunks to detect errors in a specified column using a language model.
    """
    result_dict = {}
    chunk_idx = 0
    processed_row_num = 0
    messages = [{"role": "system", "content": system_prompt}]

    for chunk in chunk_rows_by_length(rows, max_rows=max_rows):
        chunk_json = "[\n\t" + ",\n".join(chunk) + "\n]"
        processed_row_num += len(chunk)
        chunk_idx += 1
        print(f"Processing chunk {chunk_idx}: {processed_row_num} / {len(rows)} rows")

        err_prompt = gen_err_prompt(col, chunk_json)

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
                print(f"⚠️WARNING: Row index {idx} not found in samples for column '{col}'")
                continue
            else:
                result_dict[idx] = label.is_error

        if len(chunk) != len(parsed.labels):
            print(f"Warning: Chunk {chunk_idx} processed {len(parsed.labels)} labels, but expected {len(chunk)} rows.")

    return result_dict


def extract_labels(sample_idx_list: list[int], label_result: dict[int, bool]):
    """
    Extracts labels from the result dictionary based on the provided sample indices.
    """
    missing_idx = set()
    label_lst = []

    for idx in sample_idx_list:
        if idx not in label_result:
            print(f"⚠️ No label found for index {idx} — default to False")
            label_lst.append(False)
            missing_idx.add(idx)
        else:
            label_lst.append(label_result[idx])

    return label_lst, missing_idx
