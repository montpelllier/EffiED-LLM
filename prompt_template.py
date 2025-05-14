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
#     prompt = f"""
# You are a data quality expert specializing in typo detection.
#
# Given the following values from the column '{col_name}', identify which ones are valid and which are not.
#
# Return a Python list where:
# - Each item is a invalid value from the list
# - Don't include value if it is valid and correctly formatted
# - Only include value if it contains typos, inconsistencies, or unusual formatting
#
# List of values:
# {values}
#
# Important:
# - Do NOT include any explanations.
# - Do NOT wrap the list in triple backticks or any code block.
# - Respond with the list only like ['value1', 'value2', 'value3', ...]
#     """
    prompt = f"""
You are a data quality expert specializing in typo detection.

Your task is to identify and list only the values from the '{col_name}' column that clearly contain typos, spelling errors, or formatting issues.

Instructions:
- Return a Python list containing only the values that definitely contain typos or formatting inconsistencies.
- Do NOT include values that are uncommon but still plausible.
- Be strict and cautious — only mark a value as a typo if it clearly deviates from consistent patterns or conventions.
- Consider these indicators of a typo:
  - Misspellings (e.g., "Hospitl" instead of "Hospital")
  - Inconsistent or unusual casing (e.g., "new york" vs "New York")
  - Random extra symbols, characters, or whitespace
  - Misplaced abbreviations or truncations

Input values:
{values}

Output format:
- Respond with a plain Python list, like: ['wrong1', 'badEntry', 'err0r']
- Do NOT explain, justify, or wrap the list in code blocks.

Only return suspicious values. If all values seem fine, return an empty list: []
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


def build_pattern_detect_func_prompt(values: list[str]):
    prompt = f"""
You are a data format analyst. Your task is to analyze a list of string values and identify whether there are one or more consistent structural patterns among them.

Please follow these steps:

1. **Check for format patterns**, including:
   - Common prefixes, suffixes, substrings
   - Special characters if existing
   - Common word or phrase segments
   - Character counts and word positions

2. **If identifiable pattern(s) exist**, generate one or more Python functions that use re to match each unique pattern.  
   - Take a single string as input
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
def <func_name>(value):
    import re
    ...
    return <bool>


[Input Values]
{values}

Provide your functions below:
"""

    return prompt


def generate_cross_column_error_prompt(target_column: str, sample_data: str) -> str:
    prompt = f"""
You are a Python data engineer.

Your task is to write Python code (using pandas) to identify potentially incorrect values in the target column: **{target_column}**, based on patterns and relationships with 1–3 other related columns in the same dataset.

Instructions:

1. Use the provided sample data (as a list of dicts) to infer any logical or pattern-based relationships between the columns.
2. Write Python code that:
   - Loads the sample data into a pandas DataFrame.
   - Iterates through each row and checks if the value in the target column breaks expected relationships with other columns (e.g., mismatches, inconsistent patterns, bad mappings).
   - Collects any rows where the target column appears suspicious, and prints the index, value, and a brief reason.
   - If no suspicious rows are found, prints: `None found.`

3. Be strict — only flag values that clearly break observable patterns.

Requirements:
- Output only valid, executable Python code.
- Use comments in the code to explain your logic clearly.
- Focus only on validating the target column, using other columns only as reference.
- Assume the sample data is small and can be directly embedded.

[Target Column: {target_column}]
[Sample Data:]
{sample_data}
"""
    return prompt
