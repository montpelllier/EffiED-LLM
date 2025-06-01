import ast
import random
import re

import ollama
import pandas
import pandas as pd
from tqdm import tqdm

from prompt_template import build_pattern_detect_func_prompt, build_typo_detect_prompt, \
    generate_cross_column_error_prompt, build_typo_detect_prompt_v2, build_value_error_check_prompt, \
    build_cell_level_error_detection_prompt
from util import extract_functions, split_list, fix_and_parse_list, is_column_numeric, get_top_nmi_relations, \
    get_target_related_combination_freq, extract_ans_json, extract_json_list, cluster_columns_by_nmi, \
    label_cells_from_judgments, select_key_column, aggregate_consistent_predictions


def get_top_percent_values_with_avg_diff(series: pandas.Series, threshold: float = 0.9, n: float = 2.0) -> list:
    """
    获取累计占比达到指定阈值且新选项的差异超过平均差的 n 倍时终止选择。

    参数：
        series (pd.Series): 要分析的 Series。
        threshold (float): 累计比例阈值，默认是 0.8（即 80%）。
        n (float): 允许新选项的差异为平均差的最大倍数，默认是 2.0（即 2 倍）。

    返回：
        list: 值的列表，这些值的频率累计达到了指定阈值，并且差异过大时停止选择。
    """
    value_counts = series.value_counts(normalize=True)
    cumulative = value_counts.cumsum()

    # 初始化
    selected_values = []
    freq_diffs = []  # 用于记录频率差
    last_value = None
    last_freq = None
    cumulative_sum = 0  # 用来追踪累计的占比

    for value, freq in value_counts.items():
        # 累计占比更新
        cumulative_sum += freq

        # 判断是否达到累计占比阈值
        if cumulative_sum <= threshold:
            # 如果是第一个值，直接选择
            if last_value is None:
                selected_values.append(value)
                last_value = value
                last_freq = freq
            else:
                # 计算当前值和上一个值的频率差
                diff = abs(last_freq - freq)
                freq_diffs.append(diff)

                # 计算频率差的均值
                avg_diff = sum(freq_diffs) / len(freq_diffs)

                # 判断新值的差异是否大于平均差的 n 倍
                if diff > avg_diff * n:
                    break  # 如果差异过大，终止选择
                else:
                    selected_values.append(value)
                    last_value = value
                    last_freq = freq
        else:
            break  # 如果累计占比超过阈值，停止选择

    return selected_values


def get_main_values(series: pandas.Series, threshold: float = 0.8) -> list:
    """
    选择直到相邻项频率差异最大的地方停止。

    参数：
        series (pd.Series): 要分析的 Series。

    返回：
        list: 值的列表，直到频率差异最大为止。
    """
    sorted_series = series.sort_values(ascending=False)
    cumulative = sorted_series.cumsum()

    # 第一步：找出前 80% 的主流值
    main_mask = cumulative <= threshold
    main_part = sorted_series[main_mask]

    # 第二步：从剩下的部分中找最大频率差
    tail_part = sorted_series[~main_mask]
    if len(tail_part) >= 2:
        diffs = tail_part.diff().abs().dropna()
        max_diff_idx = diffs.idxmax()
        # 获取直到 max_diff_idx 这一项为止（包含它）
        cutoff_pos = tail_part.index.get_loc(max_diff_idx)
        extra_part = tail_part.iloc[:cutoff_pos]
    else:
        extra_part = pandas.Series(dtype=float)

    # 合并前80%和后面部分（最大差之前）
    final_values = main_part.index.tolist() + extra_part.index.tolist()
    return final_values


def generate_pattern_detection_functions(data: pandas.DataFrame, attribute: str) -> list:
    """
    检测数据集中是否存在模式违法。
    """
    value_cnt = data[attribute].value_counts(normalize=True)
    main_values = get_main_values(value_cnt)

    prompt = build_pattern_detect_func_prompt(main_values, col_name=attribute)
    print(prompt)
    print("======================")
    response = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
    ).response
    print(response)

    function_list = extract_functions(response)
    return function_list


def generate_typo_dict(data: pandas.DataFrame, attribute: str, verbose=False):
    values = data[attribute].value_counts().keys().tolist()
    prompt = build_typo_detect_prompt(attribute, values)

    response = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
    ).response

    if verbose:
        print(prompt)
        print("======================")
        print(response)

    try:
        typo_dict = ast.literal_eval(response)
    except Exception as e:
        print(f'Error parsing response: {response}: {e}')
        typo_dict = {}

    return typo_dict


def find_all_typo(data: pandas.DataFrame, attribute: str, verbose=False) -> set:
    values = data[attribute].value_counts().keys().tolist()
    print(f"Total values: {len(values)}")

    random.shuffle(values)
    value_groups = split_list(values, 80)

    typo_set = set()

    for i, group in tqdm(enumerate(value_groups), total=len(value_groups)):

        prompt = build_typo_detect_prompt_v2(attribute, group)
        response = ollama.generate(
            model='llama3.1:8b',
            prompt=prompt,
        ).response

        if verbose:
            print(f"\n--- Prompt Group {i + 1} ---")
            print(prompt)
            print("Response:")
            print(response)
            print("======================")

        match = re.search(r"\[[\s\S]*?]", response)
        if not match:
            print(f"[WARN] No list found in response for group {i + 1}")
            continue
        typo_list_str = match.group()

        try:
            # typo_list = json.loads(typo_list_str)
            typo_list = ast.literal_eval(typo_list_str)
        except Exception as e:
            print(f'[WARN] ast.literal_eval failed for group {i + 1}: {e}')
            print("Trying to fix and parse list...")
            print(response)
            typo_list = fix_and_parse_list(typo_list_str)

        typo_set.update(typo_list)

    return typo_set


def detect_typo(data: pandas.DataFrame, attribute: str) -> list:
    # typo_dict = generate_typo_dict(data, attribute)
    #
    # typo_vector = []
    # for value in data[attribute]:
    #     if value in typo_dict:
    #         typo_vector.append(typo_dict[value])
    #     else:
    #         print(f"Value '{value}' not found in typo_dict.")
    #         typo_vector.append(False)
    if is_column_numeric(data[attribute]):
        print(
            f"Warning: Column '{attribute}' is numeric, skip typo detection."
        )
        return [True] * len(data)

    typo_set = find_all_typo(data, attribute, False)
    print(len(typo_set), typo_set)
    typo_vector = [value not in typo_set for value in data[attribute]]

    return typo_vector


def generate_rule_detection_functions(data: pandas.DataFrame, target_column, related_columns, sample_size: int = None):
    if sample_size is None:
        sample_size = int(120 / (1 + len(related_columns)))
        # print(f'sample size set to {sample_size}')
    print(sample_size)
    related_df = data[[target_column] + related_columns].drop_duplicates()
    sampled_df = related_df.sample(n=min(sample_size, len(related_df)), random_state=42)

    sample_rows_str = '\n'.join(str(row) for row in sampled_df.to_dict(orient='records'))
    prompt = generate_cross_column_error_prompt(target_column, sample_rows_str)
    print(prompt)
    print("======================")
    response = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
    ).response
    print(response)

    function_list = extract_functions(response)
    return function_list


def get_top_frequent_values(value_counts: dict[str:float], top_k=8, min_freq=0.01, cumulative_ratio=0.6):
    if len(value_counts) <= top_k:
        return value_counts.to_dict()
    sorted_vc = value_counts[value_counts >= min_freq].sort_values(ascending=False)
    cumulative = sorted_vc.cumsum()
    main_part = sorted_vc[cumulative <= cumulative_ratio][:top_k]
    return main_part.to_dict()


def check_error(row_data, target_column, related_columns, data_name, target_value_cnts, comb_value_cnts):
    cur_value = row_data[target_column]
    if pd.isna(cur_value):
        return False

    top_freq_vals = get_top_frequent_values(target_value_cnts, top_k=8, cumulative_ratio=0.7)
    val_freq = target_value_cnts[cur_value]

    new_val = pd.to_numeric(cur_value, errors='coerce')
    val_dtype = str(new_val.dtype) if pd.notna(new_val) else 'string'

    related_dict = row_data[related_columns].to_dict()
    related_comb_freq = get_target_related_combination_freq(row_data, comb_value_cnts, target_column, related_columns,
                                                            top_k=5)

    prompt = build_value_error_check_prompt(data_name, target_column, cur_value, val_freq, val_dtype, top_freq_vals,
                                            related_dict, related_comb_freq, [], [])
    # print(prompt)
    response = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
    ).response
    # print(response)
    ans_json = extract_ans_json(response)

    is_valid = False if ans_json is None else ans_json.get('is_valid')

    if not is_valid:
        print(is_valid)
        print('====================')
        print(prompt[:1000])
        print(response)
    print(ans_json)
    return is_valid


def gen_combination_freqs(data: pandas.DataFrame, key_column: str, related_columns: list[str], sample_size: int = 10) -> \
        tuple[list[list[dict]], pandas.DataFrame]:
    valid_data = data[related_columns].dropna(how='all')

    # 生成组合频率
    freq_dataframe = valid_data.value_counts(subset=related_columns, dropna=False).reset_index(name='frequency')

    valid_keys = freq_dataframe[key_column].dropna().unique().tolist()
    sample_size = min(sample_size, len(valid_keys))
    if sample_size <= 0:
        raise ValueError("Sample size must be greater than 0.")

    keys = random.sample(valid_keys, sample_size)
    combination_freqs_lst = []

    for key in keys:
        sub_freq_dataframe = freq_dataframe[freq_dataframe[key_column] == key]
        # 把 freq_df 转换成 combination_frequency 的列表，每个元素是 dict 格式
        combination_frequency = sub_freq_dataframe.to_dict(orient='records')
        combination_freqs_lst.append(combination_frequency)
        # print(f'Total combination: {len(combo_frequency)}')

    return combination_freqs_lst, freq_dataframe


def detect_rule_violation(data, nmi_matrix, model='llama3.1:8b', sample=10, repeat=1, k=10, verbose=False):
    print(f'Detecting rule violation with model: {model}, sample: {sample}, repeat: {repeat}, top k: {k}.\n')
    column_groups = cluster_columns_by_nmi(nmi_matrix, threshold=0.4)
    label_df = pd.DataFrame(data=None, index=data.index, columns=data.columns, dtype=object)

    for column_group in column_groups.values():
        if len(column_group) < 2:
            # print(f"Column group {column_group} has less than 2 columns, skipping.")
            continue

        key_column = select_key_column(nmi_matrix, column_group)
        print(f'\nSelected key column: \'{key_column}\' from group {column_group}\n')

        combo_frequency_list, freq_df = gen_combination_freqs(data, key_column, column_group, sample_size=sample)

        for combo_frequency in combo_frequency_list:

            prompt = build_cell_level_error_detection_prompt(
                primary_key_name=key_column,
                primary_key_value=combo_frequency[0][key_column],
                columns=column_group,
                data_rows=combo_frequency,
                top_k=k
            )

            # res_pattern = r'\[\s*(?:\{[^{]*"input_id"\s*:\s*\d+[^{]*"predicted_values"\s*:\s*\[[^\]]*\][^}]*\}\s*,?\s*)+\]'
            res_pattern = r'\[\s*(?:\{\s*"row_id"\s*:\s*\d+\s*,\s*"column_judgments"\s*:\s*\{\s*(?:"[^"]+"\s*:\s*"(?:REASONABLE|SUSPICIOUS)"(?:\s*,\s*)?)+\s*\}\s*\}(?:\s*,\s*)?)+\s*\]'

            response_list = []
            for _ in range(repeat):  # N = 3~5 是常见范围
                response = ollama.generate(prompt=prompt, model=model).response
                result = extract_json_list(response, res_pattern)
                if result:
                    response_list.append(result)

            final_judgment = aggregate_consistent_predictions(response_list)
            # 生成模型结果
            # response = ollama.generate(prompt=prompt, model=model).response
            # result = extract_json_list(response, res_pattern)

            if verbose:
                print(f"\n--- Prompt for column group {column_group} ---")
                print(prompt)
                # print(f"\nModel response:\t{response}")
                # print(f"\nExtracted JSON response:\t{result}")
                print(final_judgment)

            if final_judgment is not None:
                # print(f'response length: {len(result)}, combo frequency length: {len(combo_frequency)}')
                for combo in combo_frequency:
                    print(combo)
                # print(response)
                label_df = label_cells_from_judgments(label_df, final_judgment, data, combo_frequency, key_column)
            else:
                print(f'Json extraction failed.')
                # print(response)
            #     print(combo_frequency[:10])

    return label_df