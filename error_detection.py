import ollama
import pandas

from prompt_template import generate_pattern_detect_func_prompt
from util import extract_functions, execute_function


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

    prompt = generate_pattern_detect_func_prompt(main_values)
    print(prompt)

    response = ollama.generate(
        model='llama3.1:8b',
        prompt=prompt,
    ).response
    print(response)

    function_list = extract_functions(response)
    return function_list

def detect_typo(data: pandas.DataFrame, attribute: str) -> list:
    """
    检测数据集中是否存在模式违法。
    """
    value_cnt = data[attribute].value_counts(normalize=True)
    main_values = get_main_values(value_cnt)
    return None