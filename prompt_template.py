import json

import pandas as pd


def pre_func_prompt(attr_name, data_example):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with distinguishing between clean and dirty cells in the `{attr_name}`.\n\n"

        f"Here are examples for the '{attr_name}' column:\n"
        f"{data_example}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

        "Example function code snippet:\n"
        "```python "
        f"def is_clean_[judgment](row, attr):\n"
        f"    # Value of `{attr_name}` is row[attr]\n"
        "    # Your logic here\n"
        "    return True  # or False\n"
        "```\n"
        "Provide your functions below:\n"
    )
    return prompt


def build_err_detect_prompt(attr_name: str, value_counts: dict) -> str:
    value_counts_str = "\n".join(f"{k}: {v}" for k, v in value_counts.items())

    prompt = f"""
You are a Python developer and data quality expert.

Your task is to analyze a column's values and generate Python functions that detect different types of data quality issues based on patterns, frequencies, and semantics.

### You will be given:
- A column name
- A frequency distribution of values observed in that column (value → count)

---

### Column name:
'{attr_name}'

### Frequency distribution:
{value_counts_str} 

---

### Your goals:

Write **a set of Python functions**. Each function must:
- Detect **a specific and distinct type of data quality issue**
- Be named like: `check_<column_name_lowercase>_<error_type>()`
- Take a single input parameter: `value`
- Return `True` if the value is **valid** for that rule
- Return `False` if the value contains that type of error
- Contain **no comments or explanations** in the function body

---

### Examples of common error types (not limited to these):
1. **Pattern Violations**: Format does not match expected pattern (e.g., digit structure, prefix).
2. **Missing Values**: Explicit or implicit nulls (e.g., "", "N/A", "unknown").
3. **Out-of-domain Values**: Rare values that deviate from common set (use frequency).
4. **Typos / Variants**: Spelling variants or close matches to frequent values.
5. **Length Violations**: Too short or too long based on known values.
6. **Prefix/Suffix Violations**: Unexpected beginnings or endings.
7. **Common Knowledge Violations**: Contradicts real-world expectations or logic.

---

### Output Requirements:
- Output only **Python function definitions** (1 or more)
- One function per error type — only write functions for **detectable and meaningful** issues
- Each function must be **self-contained**, **accurate**, and **not duplicated in logic**
- Do not add explanations or comments — just output valid Python code with necessary libraries and imports inside functions

Begin writing functions now.
"""
    return prompt.strip()


def build_typo_detect_prompt(col_name: str, values: list[object]) -> str:
    prompt = f"""
You are a data quality expert specializing in typo detection.

Given the following values from the column '{col_name}', identify which ones are valid and which are not.

Return a Python dictionary where:
- Each key is a value from the list
- Each value is `True` if the value is valid and correctly formatted
- Each value is `False` if it contains typos, inconsistencies, or unusual formatting

List of values:
{values}

Important:
- Do NOT include any explanations.
- Do NOT wrap the dictionary in triple backticks or any code block.
- Respond with the dictionary only — as plain text.
    """
    return prompt


def build_typo_detect_prompt_v2(col_name: str, values: list[object]) -> str:
    prompt = f"""
You are a data quality expert specializing in typo detection.

Your task is to identify and list only the values from the '{col_name}' column that may contain typos, spelling errors, or formatting issues.

Context:
- The input is a valid Python list of strings, printed directly from code.
- Your output must be a valid Python list of strings copying from input exactly.
- Your response will be parsed using Python's ast.literal_eval(). Ensure it is strictly parseable.

Instructions:
- Return a Python list containing only the values that definitely contain typos or formatting inconsistencies.
- Do NOT include values that are uncommon but still plausible.
- Even if many values appear similar, only return those that clearly violate consistent language or structure.
- Be strict and cautious — only mark a value as a typo if it clearly deviates from consistent patterns or conventions.
- Consider these indicators of a typo:
  - Misspellings
  - Inconsistent or unusual casing (e.g., "new york" vs "New York")
  - Random extra symbols, characters
  - Misplaced abbreviations or truncations

Output format:
- Respond ONLY with a valid Python list literal (e.g., ["wrong1", "badEntry", "err'0r"])
- Escape any quotes properly (e.g., `'` becomes `\\'` inside a string).
- Do NOT use single quotes or wrap the response in code blocks
- If all values are fine, return an empty list: []

Input values (from a Python print of a list):
{values}
"""
    return prompt


def generate_column_prompt(df: pd.DataFrame, sample_rows: int) -> str:
    """
    Generate a prompt for LLM to infer column-wise semantic meaning and validation rules,
    without considering relationships between columns.

    Parameters:
    - df: pd.DataFrame
    - sample_rows: int, number of rows to randomly sample for context

    Returns:
    - str: prompt ready to be fed into LLM
    """
    # Ensure the sample size is within bounds
    sampled_df = df.sample(n=min(sample_rows, len(df)), random_state=42)

    # Construct the prompt
    prompt = f"""You are a data understanding and validation assistant.

Below is a list of column names and a few sample rows from a dataset.
Your task is to infer the implicit *semantic meaning* and *validation rules* for each column,
based on the values and column names.

You are designing this for the purpose of **automated rule generation**.
These rules will be used to create Python functions that convert each row into a set of features
(e.g. out-of-range, or unexpected formats) to support unsupervised clustering or anomaly detection.

For each column:
- Explain what the column likely represents.
- Describe what types of values are expected (e.g. numeric range, categorical options, required format).
- Suggest any known or likely validation rules that could be applied.
- Be specific and concrete so the rules can be implemented as Python functions.
- Ignore relationships between columns.

Please return your output as a dictionary like this:

{{
  "column_name_1": [
    "Likely represents ...",
    "Should be ...",
    "Values should follow rule: ..."
  ],
  ...
}}

Column names:
{list(df.columns)}

Sample rows:
{sampled_df.values.tolist()}
"""
    return prompt


def build_pattern_detect_func_prompt(values: list[str], col_name: str) -> str:
    prompt = f"""
You are a data format analyst. Your task is to analyze a list of string values and identify whether there are one or more consistent structural patterns among them.

Please follow these steps:

1. **Check for format patterns**, including:
   - Common prefixes, suffixes, substrings
   - Special characters if existing
   - Common word or phrase segments
   - Character counts and word positions

2. **If identifiable pattern(s) exist**, generate one or more Python functions that use re to match each unique pattern.  
   - Accept a single DataFrame row as input
   - Each function must be precise and narrow in scope (no loose matching)
   - Each regex only contains one single prefix, suffix, or substring
   - Return True if the string matches the pattern exactly, otherwise False
   - Be named clearly based on the pattern

3. **If you think there is no consistent pattern exists**, output only: `No identifiable pattern.` instead of a function. 

Important:
- Output only valid Python functions or the single line above.
- Do not include explanations, comments, or usage examples.
- Do not combine unrelated formats into one expression.
- Each function must independently import re inside the function body.

Output format:
Pattern 1: <pattern_1>>
description: <description> 
def <func_name>(row):
    import re
    ... 
    return <bool>

[Column Name: {col_name}]
[Input Values]
{values}

Provide your functions below:
"""

    return prompt


def generate_cross_column_error_prompt(target_column: str, sample_data: str) -> str:
#     prompt = f"""
# You are a Python data engineer.
#
# Your task is to write Python code (using pandas) to identify potentially incorrect values in the target column: **{target_column}**, based on patterns and relationships with 1–3 other related columns in the same dataset.
#
# Instructions:
#
# 1. Use the provided sample data (as a list of dicts) to infer any logical or pattern-based relationships between the columns.
# 2. Write Python code that:
#    - Loads the sample data into a pandas DataFrame.
#    - Iterates through each row and checks if the value in the target column breaks expected relationships with other columns (e.g., mismatches, inconsistent patterns, bad mappings).
#    - Collects any rows where the target column appears suspicious, and prints the index, value, and a brief reason.
#    - If no suspicious rows are found, prints: `None found.`
#
# 3. Be strict — only flag values that clearly break observable patterns.
#
# Requirements:
# - Output only valid, executable Python code.
# - Use comments in the code to explain your logic clearly.
# - Focus only on validating the target column, using other columns only as reference.
# - Assume the sample data is small and can be directly embedded.
#
# [Target Column: {target_column}]
# [Sample Data:]
# {sample_data}
#
# Example function code snippet:
# ```python
# def is_clean_[judgment](row, attr):
#     # Value of `{target_column}` is row[attr]
#     # Your logic here
#     return True  # or False
# ```
#
# Provide your functions below:
# """
    prompt = f"""
You are a Python data engineer.

Your task is to write one or more Python functions that detect whether values in the target column '{target_column}' are **potentially incorrect**, using only logical relationships and consistency rules involving other columns in the same row.

Instructions:

1. You will be provided a pandas DataFrame row (as input to your function).
2. Each function must:
    - Accept a single DataFrame row as input
    - Focus solely on validating the correctness of the value in the target column
    - Use other columns only as reference for context
    - Return True if the target column's value is logically consistent and acceptable
    - Return False only if the target column’s value is **clearly** invalid based on one or more of the following:
        • Inter-column relationships (e.g., code or label mismatches)
        • Logical inconsistency (e.g., invalid combinations of values)
        • Common sense rules (e.g., geographic mismatches or structural anomalies)

3. Guidelines:
    - Do NOT validate other columns directly — only use them to assess the {target_column} column
    - Do NOT return or print any output — just define the function(s)
    - Function names should follow the pattern: is_<error_type>_in_{target_column}
    - Only write functions when there is a meaningful check to implement
    - If no reasonable rule can be inferred for the target column, output nothing

[Target Column: {target_column}]
[Sample Data:]
{sample_data}
[Output Format:]
def <function_name>(row):
    import <library>
    return <bool>
"""
    return prompt


def build_cell_validation_prompt(
    dataset_name: str,
    col_name: str,
    value: str,
    val_dtype: str,
    val_freq: int,
    top_freq_vals: list,
    related_values: dict,
    positive_vals: list,
    negative_vals: list
) -> str:
    prompt = f"""
You are a data quality validation expert.

Your task is to determine whether a specific value in a dataset is potentially incorrect, based on contextual, statistical, and semantic information.

Dataset: {dataset_name}
Column: {col_name}
Value to evaluate: {repr(value)}

You are provided the following information to help your judgment:

1. **Value Type (dtype)**: {val_dtype}
2. **Frequency of this value in the column**: {val_freq}
3. **Top frequent values in the column**: {top_freq_vals}
4. **Values in related columns from the same row**: {related_values}
5. **Known valid values (positives)**: {positive_vals if positive_vals else "None"}
6. **Known invalid values (negatives)**: {negative_vals if negative_vals else "None"}

Instructions:
- Use common sense, domain knowledge, frequency patterns, and inter-column relationships to evaluate the value.
- Focus on internal consistency (e.g. spelling, format, category misalignment).
- Consider whether the value is plausible or breaks obvious patterns.
- Consider whether the related values suggest the target value is suspicious.

Response format:
- Return `True` if the value is likely incorrect or suspicious.
- Return `False` if the value seems correct or plausible in context.
- Do NOT include any explanations or extra output.
"""
    return prompt.strip()


def build_value_error_check_prompt(
        dataset_name: str,
        col_name: str,
        value: str,
        val_freq: float,
        val_dtype: str,
        top_freq_vals: list,
        related_values: dict,
        comb_freqs: list[dict[str, object]],
        positive_vals: list[str],
        negative_vals: list[str]
) -> str:
    """
    Build a JSON-style prompt to ask an LLM whether a value in a specific column is likely incorrect,
    based on frequency, top values, related fields, and examples.

    Returns:
        A string prompt to send to the LLM.
    """

    prompt_dict = {
        "task": (
            "Determine if the provided value in the specified column is likely incorrect, inconsistent, or a typo, "
            "based on its frequency, its context in other fields, and known common/rare value patterns in the dataset."
        ),
        "dataset_name": dataset_name,
        "column_name": col_name,
        "value_to_check": value,
        "value_frequency": val_freq,
        "value_dtype": val_dtype,
        "top_frequent_values_in_column": top_freq_vals,
        "related_column_values": related_values,
        "related_value_combination_frequencies": comb_freqs,
        "known_positive_values": positive_vals,
        "known_negative_values": negative_vals,
         "chain_of_thought_instructions": [
            "Step 1: Check if the input value looks similar in spelling or form to the most frequent values in the column.",
            "Step 2: Compare it against known positive and negative values. Does it match the characteristics of known correct or incorrect entries?",
            "Step 3: Evaluate the values in related fields — but keep in mind these values might also contain errors.",
             "Step 4: Use the related_value_combination_frequencies to assess how often the value_to_check appears with the related_column_values:",
             "    - Look for frequent and consistent combinations. If a value_to_check appears frequently (e.g., count > 10) with related values, it's likely valid.",
             "    - If a combination is rare or unique (e.g., count = 1), investigate further. But do NOT penalize a valid value just because related values might be misspelled.",
             "    - If there are multiple records where value_to_check appears with clean and consistent related values, that supports its validity.",
             "    - If the same value_to_check appears with many misspelled related values, the issue is likely in the related columns, not the value itself.",
            "Step 5: Consider possible typo patterns in the value_to_check (e.g., swapped letters, truncation, phonetic errors).",
            "Step 6: Make a final judgment: Is the value_to_check likely correct and consistent based on frequency, combination patterns, and surrounding context?",
            "    - Only mark it as incorrect if the value_to_check itself is clearly wrong (e.g., typo, implausible, or inconsistent). Do not penalize based on unrelated or potentially incorrect related values."        ],
         "output_instruction": (
            "Respond step by step with reasoning.\n"
            "Then return a single JSON object with this structure:\n"
            "{\n"
            "  \"is_valid\": true/false,\n"
            "  \"reason\": \"brief explanation of the judgement\",\n"
            "  \"value\": \"the input value\"\n"
            "}\n"
            "Return this JSON, and ensure it is valid and parseable."
        )
    }

    return json.dumps(prompt_dict, ensure_ascii=False, indent=2)


def build_multicol_typo_check_prompt(
        dataset_name: str,
        column_names: list[str],
        value_combinations: list[dict[str, object]]
) -> str:
    """
    Build a prompt asking LLM to check for typos or incorrect values in each field of multiple value combinations.

    Args:
        dataset_name: Name of the dataset.
        column_names: The list of columns that appear in the combinations.
        value_combinations: List of value combination dictionaries. Each dict has column-value pairs and a 'count' key.

    Returns:
        A string prompt in JSON format ready for LLM input.
    """
    prompt_dict = {
        "task": "Check each value in the following value combinations for potential typos, misspellings, or formatting errors.",
        "dataset": dataset_name,
        "columns": column_names,
        "value_combinations_with_frequency": value_combinations,
        "instructions": [
            "You are given several combinations of values from a table. Each combination includes values from multiple columns and their frequency in the dataset.",
            "Your task is to evaluate each combination and determine if any of the individual values in the columns are likely to be incorrect.",
            "For each value, consider:",
            "- Does it match common spelling or formatting conventions?",
            "- Is it similar to values seen in other combinations?",
            "- Does it look like a typo (e.g., character swaps, phonetic errors, casing issues, random letters)?",
            "- Does it only appear in combinations with very low frequency?",
            "Do NOT mark a value as wrong just because it is rare. Only mark it if it clearly seems inconsistent.",
            "Output a JSON object listing each combination and for each column, indicate whether the value looks valid or not, with a brief reason.",
        ],
        "output_format": {
            "combinations_evaluation": [
                {
                    "combination_index": 0,
                    "evaluation": {
                        "ColumnName1": {"is_valid": True, "reason": "Looks consistent", "value": "value1"},
                        "ColumnName2": {"is_valid": False, "reason": "Typo in hospital name", "value": "value2"},

                    }
                },
                {
                    "combination_index": 1,
                },
            ]
        }
    }

    return json.dumps(prompt_dict, ensure_ascii=False, indent=2)


def build_prediction_prompt(
    target_column: str,
    related_columns: list,
    related_values: dict,
    combo_frequencies: list,
    max_combos: int = 10
) -> str:
    """
    构造自然语言 prompt，用于 LLM 预测 target_column 的最可能值。

    参数:
    - target_column: 目标列名，例如 'postal_code'
    - related_columns: 相关列名列表，例如 ['province', 'city', 'area_code']
    - related_values: 当前待判断行的相关列值，如 {'province': 'Guangdong', 'city': 'Shenzhen', 'area_code': '0755'}
    - combo_frequencies: 已知的高频组合记录，每条为 dict，包含所有列及 'frequency' 字段
    - max_combos: 限制最多展示多少条组合示例

    返回:
    - prompt: 可供 LLM 使用的自然语言提示词
    """

    prompt_parts = ["You are given a structured dataset with known value combinations and their relative frequencies.",
                    f"The goal is to predict the most likely value for the column '{target_column}', given values in other related columns.\n",
                    "The current row has the following values:"]


    # 当前行相关值
    for col in related_columns:
        prompt_parts.append(f"- {col}: {related_values.get(col, 'UNKNOWN')}")

    prompt_parts.append("\nHere are known historical combinations and their relative frequencies:")

    # 限制最多展示 N 条组合
    top_combos = combo_frequencies[:max_combos]
    for combo in top_combos:
        parts = [f"{k}={v}" for k, v in combo.items() if k != "frequency"]
        parts_str = ", ".join(parts)
        prompt_parts.append(f"- [{parts_str}] → freq: {combo['frequency']:.3f}")

    # 提出问题
    prompt_parts.append(f"\nBased on these examples, what is the most likely value for '{target_column}'?\n"
                        f"Please return only the value, without explanation.")

    return "\n".join(prompt_parts)

def build_batch_prediction_prompt(
    target_column: str,
    related_columns: list[str],
    input_rows: list[dict],
    combo_frequencies: list[dict],
    max_combos: int = 10
) -> str:
    """
    构造自然语言提示词，让 LLM 对多组输入相关值，预测统一目标列的最可能值。

    参数:
    - target_column: 需要预测的列名，如 'postal_code'
    - related_columns: 与其相关的列名，如 ['province', 'city', 'area_code']
    - input_rows: 多组相关列值，列表形式，每个元素为 dict，如 {'province': 'Guangdong', 'city': 'Shenzhen'}
    - combo_frequencies: 已知的高频组合样本，每项为 dict，包含所有列及 'frequency'
    - max_combos: 显示的历史组合最大数量

    返回:
    - prompt: 自然语言提示词字符串
    """

    prompt_parts = [
        "You are given a structured dataset with known combinations of column values and their frequencies.",
        f"The goal is to predict the most likely value for the target column: '{target_column}', based on related fields.\n",
        "Below are some high-frequency historical combinations from the dataset:"
    ]

    top_combos = combo_frequencies[:max_combos]
    for combo in top_combos:
        parts = [f"{k}={v}" for k, v in combo.items() if k != "frequency"]
        parts_str = ", ".join(parts)
        prompt_parts.append(f"- [{parts_str}] → freq: {combo['frequency']}")

    prompt_parts.append("\nNow, for each of the following inputs, predict the most likely value for the target column.")

    for i, row in enumerate(input_rows):
        parts = [f"{col}: {row.get(col, 'UNKNOWN')}" for col in related_columns]
        prompt_parts.append(f"{i + 1}. " + ", ".join(parts))

    prompt_parts.append(
        "\nRespond with a JSON array, where each object contains:\n"
        "- `input_id`: index of the input (starting from 1),\n"
        "- `predicted_value`: the most likely value for the target column.\n"
        "Format:\n[\n"
        "  {\"input_id\": 1, \"predicted_value\": \"value1\"},\n"
        # "  {\"input_id\": 2, \"predicted_value\": \"value2\"},\n"
        "  ...\n"
        "]\n"
        "Only return the JSON array, no explanation."
    )

    return "\n".join(prompt_parts)
