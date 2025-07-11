import random

import ollama

from detection.prompt_templates import *
from detection.tmp_util import chunk_rows_by_length
from evaluation.evaluation import evaluate_model, evaluate_column_predictions
from features import *
from model import *



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

mistral_7b = 'mistral:7b'
deepseek_r1_8b = 'deepseek-r1:8b'
llama3_8b = 'llama3.1:8b'
llama3_3b = 'llama3.2:3b'
qwen3_4b = 'qwen3:4b'
qwen3_8b = 'qwen3:8b'
gemma3n_e4b = 'gemma3n:e4b'
gemma3_4b = 'gemma3:4b'
phi3_8b = 'phi3:3.8b'

model_name = qwen3_4b
use_thinking = False

print(f"Using model: {model_name}")

data_clean = pd.read_csv(f'../data/{dataset_name}_clean.csv', dtype=str)
data_error = pd.read_csv(f'../data/{dataset_name}_error-01.csv', dtype=str)

err_labels = data_clean != data_error

pred_df = DataFrame(np.zeros_like(data_error, dtype=bool), columns=data_error.columns)

cluster_params = {
    't': 0.4,
    'criterion': 'distance',
}

max_input_chars = 400

for col in data_error.columns:
    print("---" * 50)

    feature_df = generate_features(data_error, col)

    unique_cnts = data_error[col].nunique()
    print(f"Unique values in column '{col}': {unique_cnts}")

    tmp_clusters, _ = cluster_features(feature_df, cluster_params=cluster_params)
    tmp_cluster_nums = len(np.unique(tmp_clusters))
    max_clusters = min(int(unique_cnts), 500, tmp_cluster_nums)

    col_cluster_params = {
        't': max_clusters,
        'criterion': 'maxclust',
    }
    clusters, samples = cluster_features(feature_df, cluster_params=col_cluster_params)

    sample_data = data_error.iloc[samples]
    sample_val_lst = sample_data[col].tolist()

    rows = [{"index": int(idx), col: val} for idx, val in zip(samples, sample_val_lst)]

#     sys_prompt = """
# You are an expert error detection assistant. Your core task is to meticulously examine individual values within a dataset and determine if they contain any errors.
# An error is broadly defined and includes, but is not limited to, typos, grammatical mistakes, formatting inconsistencies (e.g., incorrect date formats, unexpected characters), or logical inconsistencies when compared to other related data points within the same record or common knowledge.
# """
    sys_prompt = """
You are an error detection assistant. Your job is to label whether each value contains an error.
An error could be a typo or a formatting issue.

Respond ONLY in valid JSON.
"""

    messages = [
        {
            "role": "system",
            "content": sys_prompt,
        },
    ]

    result_dict = {}
    missing_indices = set()
    sample_labels = []
    chunk_idx = 0

    processed_row_num = 0
    for chunk in chunk_rows_by_length(rows, max_chars=max_input_chars):
        # 构造 JSON 输入字符串
        # print(chunk)
        chunk_json = "[\n" + ",\n".join(chunk) + "\n]"

        processed_row_num += len(chunk)
        print(f"Processing chunk {chunk_idx}: {processed_row_num} / {len(rows)} rows")
        chunk_idx += 1

        # print(f"Prompt:\n{prompt}\n")
        messages.append({"role": "user", "content": gen_err_prompt(col, chunk_json)})
        # print(messages[-1])

        response = ollama.chat(
            model=model_name,
            messages=messages,
            format=LabelList.model_json_schema(),
            think=use_thinking,
            options={"temperature": 0},
        ).message

        json_str = extract_label_list_json(response.content)
        parsed = LabelList.model_validate_json(json_str)
        # print(parsed)

        messages.append({"role": "assistant", "content": json_str})

        for label in parsed.labels:
            if label.index not in samples:
                print(f"⚠️ Index {label.index} not found in samples for column '{col}'")
                continue

            result_dict[label.index] = label.is_error

        if len(chunk) != len(parsed.labels):
            print(f"Warning: Chunk {chunk_idx} processed {len(parsed.labels)} labels, but expected {len(chunk)} rows.")

    for idx in samples:
        if idx not in result_dict:
            print(f"⚠️ No label found for index {idx} — default to False")
            sample_labels.append(False)
            missing_indices.add(idx)
        else:
            sample_labels.append(result_dict[idx])

    if missing_indices:
        print(f"⚠️ Missing labels for indices: {missing_indices}")
        retry_rows = [{"index": idx, col: data_error.iloc[idx][col]} for idx in missing_indices]
        retry_batches = list(chunk_rows_by_length(retry_rows, max_chars=max_input_chars))

        for retry_chunk in retry_batches:
            retry_json = "[\n" + ",\n".join(retry_chunk) + "\n]"

            retry_prompt = f"""
Retrying missing values for column `{col}`.
Input:
{retry_json}
""".strip()

            messages.append({"role": "user", "content": retry_prompt})

            retry_response = ollama.chat(
                model=model_name,
                messages=messages,
                format=LabelList.model_json_schema(),
                think=use_thinking,
                options={"temperature": 0},
            ).message

            json_str = extract_label_list_json(retry_response.content)
            retry_parsed = LabelList.model_validate_json(json_str)

            for label in retry_parsed.labels:
                if label.index not in samples:
                    print(f"⚠️ Retry index {label.index} not found in samples for column '{col}'")
                    continue

                result_dict[label.index] = label.is_error
                missing_indices.discard(label.index)

    real_labels = err_labels.iloc[samples][col].tolist()
    result = evaluate_column_predictions(real_labels, sample_labels)
    print(f"LLM prediction results for column '{col}': {result}")

    for idx, label in zip(samples, sample_labels):
        real_label = err_labels.iloc[idx][col]
        if label == real_label:
            continue
        print(f"predicted label *{label}* does not match actual label *{real_label}* for index {idx}")
        data = data_error.loc[idx, col]
        print(data)

    prediction = propagate_labels(clusters, samples, sample_labels)
    pred_df[col] = prediction

evaluate_model(err_labels, pred_df)
