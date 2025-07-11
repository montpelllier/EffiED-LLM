
def gen_err_prompt(column_name, data_json):
    """
    创建用于检测拼写错误的提示模板
    """
    prompt = f"""
You are given a list of data row extracted from a dataset, where the target column named `{column_name}`. 
Label each value from the target column as either an error or not an error.

- Each value has an "index" field, which is the original index from the dataset.
- Do **not** use your own internal numbering (e.g., 0, 1, 2, etc.). Use **only the "index" values provided**.
- Only label the values in the target column, do not label other columns.
- Only label the values that you are confident about, do not label the values that you are not sure about.

Input:
{data_json}
"""
    return prompt

def gen_fd_prompt(column_name, data_json):
    """
    创建用于检测拼写错误的提示模板
    """
    prompt = f"""
You are given a list of data row extracted from a dataset. The target column is named `{column_name}`.
Label each value of the target column as either an error or not an error.

- Each row has an "index" field, which is the original index from the dataset.
- **DO NOT** use your own internal numbering (e.g., 0, 1, 2, etc.). Use **only the "index" values provided**.

Input:
{data_json}
"""
    return prompt

def gen_anno_prompt(column_name, data_json, column_annoations):
    """
    创建用于检测拼写错误的提示模板
    """
    prompt = f"""
You are given a list of values extracted from a dataset column named `{column_name}`.
Label each value of the given column as either an error or not an error. An error could be a typo, a formatting issue, or any other kind of data quality problem.
Use column description help you determine if the value is an error. 

- Each row has an "index" field, which is the original index from the dataset.
- **Never** use your own internal numbering (e.g., 0, 1, 2, etc.). Use **only the "index" values provided**.

Column annotations:
{column_annoations}

Input:
{data_json}
"""
    return prompt


def gen_metadata_prompt(sample_data_str):
    prompt = f"""
    You are given a table sample from a dataset. This dataset may contain noisy or incorrect values, but your goal is to infer the intended structure and semantics.

    Your task is to generate the following metadata for EVERY column in the table:
    1. A brief, natural-language description of the column. Include the rules or patterns you infer from column values, related columns, and the column name.
    2. The most likely corresponding Schema.org property (if applicable) based on given example values.
    3. The ideal or most appropriate data type (e.g., string, date, float, category).
    4. A list of top 3 related columns. 

    Here are a few example rows from the dataset:
    {sample_data_str}
    """
    return prompt


