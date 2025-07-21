import random

from dataset import Dataset
from detection.tmp_util import process_data_chunks, extract_labels
from evaluation.evaluation import evaluate_model, evaluate_column_predictions
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

mistral_7b = 'mistral:7b'
deepseek_r1_8b = 'deepseek-r1:8b'
llama3_8b = 'llama3.1:8b'
llama3_3b = 'llama3.2:3b'
qwen_2d5_7b = 'qwen2.5:7b'
qwen3_4b = 'qwen3:4b'
qwen3_8b = 'qwen3:8b'
gemma3n_e4b = 'gemma3n:e4b'
gemma3_4b = 'gemma3:4b'
phi3_8b = 'phi3:3.8b'

model_name = qwen3_4b
use_thinking = False
# method = "chat"
method = "generate"
max_rows = 10
print(f"Using model: {model_name}, thinking mode: {use_thinking}, method: {method}")

datset = Dataset(dataset_name)
data_error = datset.dirty_data
err_labels = datset.error_labels
pred_df = DataFrame(np.zeros_like(data_error, dtype=bool), columns=data_error.columns)

cluster_params = {
    'n_clusters': 20,
}
cluster_df, repre_df = cluster_dataset(data_error, cluster_params=cluster_params)
print(repre_df.head(3))

for col in data_error.columns:
    print("---" * 50)
    unique_cnts = data_error[col].nunique()
    print(f"Unique values in column '{col}': {unique_cnts}")

    col_cluster = cluster_df[col]
    col_repre = repre_df[col].astype(int).tolist()

    sample_data = data_error.iloc[col_repre]
    sample_val_lst = sample_data[col].tolist()

    rows = [{"row_id": int(idx), col: val} for idx, val in zip(col_repre, sample_val_lst)]
    sys_prompt = """
You are an error detection assistant. Your task is to label whether each value contains an error.
An error could be a typo or a formatting issue.

Respond ONLY in valid JSON.
"""

    result_dict = process_data_chunks(
        rows,
        col,
        sys_prompt,
        model_name,
        col_repre,
        use_thinking=use_thinking,
        max_rows=max_rows,
        method=method
    )

    sample_labels, missing_indices = extract_labels(col_repre, result_dict)

    real_labels = err_labels.iloc[col_repre][col].tolist()
    result = evaluate_column_predictions(real_labels, sample_labels)
    print(f"LLM prediction results for column '{col}': {result}")

    for idx, label in zip(col_repre, sample_labels):
        real_label = err_labels.iloc[idx][col]
        if label != real_label:
            print(
                f"predicted label *{label}* does not match actual label *{real_label}* for index {idx}: {data_error.loc[idx, col]}")

    prediction = propagate_labels(col_cluster, col_repre, sample_labels)
    pred_df[col] = prediction

evaluate_model(err_labels, pred_df)
