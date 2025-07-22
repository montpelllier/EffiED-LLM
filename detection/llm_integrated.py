import logging
import os
import random
import sys

from detection.prompt_templates import gen_fewshot_prompt
from init import *
from detection.tmp_util import process_data_chunks, extract_labels, generate_data_rows, select_few_shot_examples
from evaluation.evaluation import evaluate_model, evaluate_column_predictions
from features import *
from util import compute_nmi_matrix, get_top_nmi_relations

random.seed(42)

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# dataset_name = 'flights'
# dataset_name = 'movies_1'
# dataset_name = 'billionaire'
# dataset_name = 'beers'
# dataset_name = 'hospital'
dataset_name = 'rayyan'

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

model_name = llama3_8b
use_thinking = False
few_shot = 0
method = "chat"
# method = "generate"
max_rows = 10
use_attr = True
improvement = None

cluster_params = {
    'n_clusters': 20,
}

logger = get_logger({
    'model_name': model_name,
    'dataset_name': dataset_name,
    'use_thinking': use_thinking,
    'few_shot': few_shot,
    'use_attr': use_attr,
    'method': method,
    'max_rows': max_rows
})

# -----------------------------------------
logger.info(f"Using model: {model_name}, thinking mode: {use_thinking}, {few_shot}-shot, method: {method}")

datset = Dataset(dataset_name)
data_error = datset.dirty_data
err_labels = datset.error_labels
pred_df = DataFrame(np.zeros_like(data_error, dtype=bool), columns=data_error.columns)

cluster_df, repre_df = cluster_dataset(data_error, cluster_params=cluster_params)
logger.info(repre_df.head(3))

column_num = len(data_error.columns)
nmi_matrix = compute_nmi_matrix(data_error)
related_col_dict = get_top_nmi_relations(nmi_matrix, max_attr=3, min_attr=1)
for item in related_col_dict.items():
    col, related_cols = item
    logger.info(f"Column '{col}' related to: {related_cols}")

for col in data_error.columns:
    logger.info("---" * 50)
    unique_cnts = data_error[col].nunique()
    logger.info(f"Unique values in column '{col}': {unique_cnts}")

    col_cluster = cluster_df[col]
    col_repre = repre_df[col].astype(int).tolist()

    few_shot_examples = select_few_shot_examples(data_error, err_labels, col, num_examples=few_shot, strategy='balanced')
    few_shot_prompt = gen_fewshot_prompt(col, few_shot_examples)
    logger.info(f"Few-shot examples for column '{col}': {few_shot_prompt}")

    if use_attr:
        related_cols = related_col_dict[col]
        logger.info(f"Processing column '{col}' with related columns: {related_cols}")
        column_lst = [col] + related_cols
    else:
        column_lst = [col]

    rows = generate_data_rows(data_error, column_lst, col_repre)

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
        fewshot_prompt=few_shot_prompt,
        max_rows=max_rows,
        method=method,
        logger=logger
    )

    sample_labels, missing_indices = extract_labels(col_repre, result_dict, logger)

    real_labels = err_labels.iloc[col_repre][col].tolist()
    result = evaluate_column_predictions(real_labels, sample_labels)
    logger.info(f"LLM prediction results for column '{col}': {result}")

    for idx, label in zip(col_repre, sample_labels):
        real_label = err_labels.iloc[idx][col]
        if label != real_label:
            logger.info(
                f"predicted label *{label}* does not match actual label *{real_label}* for index {idx}: {data_error.loc[idx, col]}")

    prediction = propagate_labels(col_cluster, col_repre, sample_labels)
    pred_df[col] = prediction

evaluate_model(err_labels, pred_df, logger)
