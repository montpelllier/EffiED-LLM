import random
import re
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import test2
from evaluation.evaluation import evaluate_model
from features import *
from llm_client import OllamaClient

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

data_clean = pd.read_csv(f'../data/{dataset_name}_clean.csv', dtype=str)
data_error = pd.read_csv(f'../data/{dataset_name}_error-01.csv', dtype=str)

err_labels = data_clean != data_error

ds = 'deepseek-r1:8b'
llama3 = 'llama3.1:8b'
llama3_3b = 'llama3.2:3b'
qwen3 = 'qwen3:4b'
llm = OllamaClient(model_name=llama3)

pred_df = DataFrame(np.zeros_like(data_error, dtype=bool), columns=data_error.columns)


class ErrorLabel(BaseModel):
    value: str
    is_typo: bool

class ErrLabelList(BaseModel):
    labels: list[ErrorLabel]


for col in data_error.columns:
    print(f"Processing column: {col}")
    feature_df = generate_features(data_error, col)
    clusters, samples = cluster_features(feature_df)
    sample_data = data_error.iloc[samples]
    sample_val_lst = sample_data[col].tolist()
    sample_vals = set(sample_val_lst)  # 去重
    print(f"sample values: {sample_vals}")

    prompt = f"""
You are a typo detection assistant.

You are given a list of unique values extracted from a dataset column named `{col}`. Some of these values may contain typos or spelling mistakes. Your task is to identify which values are likely typos.

Please response **a valid Python dictionary** text directly in the following format:
{{value1: True, value2: False, ...}}

- Use **True** if the value is likely a typo or misspelling.
- Use **False** if the value looks normal or correctly spelled.
- Do **not** return any explanations or extra text, only the dictionary text.

Column name :`col`
Values to evaluate:
{sample_vals}
    """.strip()
    # print(f"Prompt:\n{prompt}\n")
    print("Generating response...")

    # response = ""
    # for chunk in llm.generate_stream(prompt=prompt):
    #     response += chunk
    #     print(chunk, end='', flush=True)
    # print("\nDone generating response.\n")
    response = llm.generate(
        prompt=prompt,
        thinking=True,
        show_thinking=False
    )
    print(response)

    cleaned_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    result_dict = test2.extract_result_dict(cleaned_text)

    print(f"Extracted dictionary: {result_dict}")

    # Cluster 标签传播逻辑
    sample_labels = []
    for idx, sample_idx in enumerate(samples):
        cluster_id = idx + 1
        label = result_dict.get(data_error[col].iloc[sample_idx])
        sample_labels.append(label)

    prediction = propagate_labels(clusters, samples, sample_labels)
    pred_df[col] = prediction

evaluate_model(err_labels, pred_df)