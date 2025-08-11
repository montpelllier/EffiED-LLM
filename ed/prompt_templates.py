def gen_err_prompt(column_name: str, data_json, fewshot_str=None, rule_str=None):
    """
    Prompt template for error detection in a specific column of a dataset.
    """
    prompt = f"""
Instructions:
- You are given a list of data rows extracted from a dataset, where the target column named `{column_name}`. Label each value from the target column as either an error or not an error. An error could be a typo or a formatting issue.
- Only evaluate the `{column_name}` value.
- Only label values you are confident about. If you are not sure, do not label it as an error.
"""
    if rule_str:
        prompt += "- If value does not follow the rules, label it as an error.\n"
        prompt += rule_str

    if fewshot_str:
        prompt += fewshot_str

    prompt += f"""
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


def gen_fewshot_prompt(column, examples):
    """
    Generates a few-shot prompt for error detection based on sample data.
    """
    if not examples or len(examples) == 0:
        return None

    exclude_keys = ['clean_value']

    few_shot_prompt = "\n\nHere are some examples:\n\n"

    for i, example in enumerate(examples, 1):
        row_data = {k: v for k, v in example.items() if k not in exclude_keys}
        origin_val = example[column]
        clean_val = example['clean_value']

        few_shot_prompt += f"Example {i}:\n"
        few_shot_prompt += f"Row: {row_data}\n"
        few_shot_prompt += f"Column: {column}\n"
        few_shot_prompt += f"Original value: {origin_val}\n"
        few_shot_prompt += f"Corrected value: {clean_val}\n\n"

    return few_shot_prompt


def gen_rule_prompt(column, rule):
    """
    Generates a prompt for error detection based on rules.
    """
    if not rule:
        return None

    rule_prompt = f"\n\nRules for column '{column}':\n"
    for key, value in rule.items():
        rule_prompt += f"{key}: {value}\n"

    return rule_prompt
