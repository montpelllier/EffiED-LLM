import random

import ollama
from pydantic import BaseModel

from evaluation.evaluation import evaluate_model
from features import *

random.seed(42)

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# dataset_name = 'flights'
# dataset_name = 'movies'
# dataset_name = 'billionaire'
# dataset_name = 'beers'
dataset_name = 'hospital'
# dataset_name = 'rayyan'
# dataset_name = 'tax50k'

ds_8b = 'deepseek-r1:8b'
llama3_8b = 'llama3.1:8b'
llama3_3b = 'llama3.2:3b'
qwen3_4b = 'qwen3:4b'

model_name = llama3_8b

data_clean = pd.read_csv(f'../data/{dataset_name}_clean.csv', dtype=str)
data_error = pd.read_csv(f'../data/{dataset_name}_error-01.csv', dtype=str)

err_labels = data_clean != data_error

pred_df = DataFrame(np.zeros_like(data_error, dtype=bool), columns=data_error.columns)


class ErrorLabel(BaseModel):
    index: int
    value: str
    is_typo: bool


class LabelList(BaseModel):
    labels: list[ErrorLabel]


for col in data_error.columns:
    print("-" * 50)
    print(f"Processing column: {col}")
    feature_df = generate_features(data_error, col)
    clusters, samples = cluster_features(feature_df)
    sample_data = data_error.iloc[samples]
    sample_val_lst = sample_data[col].tolist()

    # sample_vals = set(sample_val_lst)  # 去重
    # print(f"sample values: {sample_vals}")

    rows = [{"index": int(idx), "value": val} for idx, val in zip(samples, sample_val_lst)]
    print(rows)
    #     prompt = f"""
    # You are a typo detection assistant.
    #
    # You are given a list of unique values extracted from a dataset column named `{col}`.
    # Each value is associated with an index from the dataset.
    # Some values may contain typos or spelling mistakes. Your task is to return whether each is a typo.
    #
    # Please response **a valid Python dictionary** text directly in the following format:
    # {{value1: True, value2: False, ...}}
    #
    # - Use **True** if the value is likely a typo or misspelling.
    # - Use **False** if the value looks normal or correctly spelled.
    # - Do **not** return any explanations or extra text, only the dictionary text.
    #
    # Column name :`col`
    # Values to evaluate:
    # {rows}
    #     """.strip()
    prompt = f"""
    You are a typo detection assistant.

    You are given a list of unique values extracted from a dataset column named `{col}`. 
    Each value is associated with an index from the dataset. 
    Some values may contain typos or spelling mistakes. Your task is to return whether each is a typo.

    Please return the result in **valid JSON** format directly.

    - Use **True** if the value is likely a typo or misspelling.
    - Use **False** if the value looks normal or correctly spelled.

    Input values:
    {rows}
        """.strip()
    # print(f"Prompt:\n{prompt}\n")
    print("Generating response...")

    response = ollama.generate(
        model=llama3_8b,
        prompt=prompt,
        format=LabelList.model_json_schema(),
        options={"temperature": 0},
    ).response

    parsed = LabelList.model_validate_json(response)
    print(parsed)

    result_dict = {label.index: label.is_typo for label in parsed.labels}
    # print(f"Extracted dictionary: {result_dict}, {samples}")

    sample_labels = []
    for rep_idx in samples:
        if rep_idx not in result_dict:
            print(f"Warning: No label found for index {rep_idx} in column '{col}'")
            sample_labels.append(False)
        else:
            sample_labels.append(result_dict[rep_idx])
    print(sample_labels)

    prediction = propagate_labels(clusters, samples, sample_labels)
    pred_df[col] = prediction

evaluate_model(err_labels, pred_df)
