import re
from collections import defaultdict
from itertools import combinations

import numpy
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


# def encode(series):
#     return LabelEncoder().fit_transform(series.astype(str))
#
#
# def compute_nmi(col1, col2):
#     mask = col1.notna() & col2.notna()
#     col1, col2 = col1[mask], col2[mask]
#     encoded_co1, encoded_co2 = encode(col1), encode(col2)
#     return normalized_mutual_info_score(encoded_co1, encoded_co2, average_method='arithmetic')
#
#
# def compute_all_nmi(dataframe):
#     dataframe = dataframe.dropna(axis=1, how='all')  # 去除全为空的列
#     results = {}
#     for col1, col2 in combinations(dataframe.columns, 2):
#         nmi = compute_nmi(dataframe[col1], dataframe[col2])
#         results[(col1, col2)] = nmi
#     return results
#
#
# def cal_strong_res_column_nmi(nmi_results, rel_top=1, threshold=0):
#     results = defaultdict(dict)
#     for (col1, col2), nmi in nmi_results.items():
#         if nmi > threshold:
#             results[col1][col2] = nmi
#             results[col2][col1] = nmi
#     # Select the top 2 col2 of col1 considering nmi
#     top_results = defaultdict(dict)
#     for col1, related_cols in results.items():
#         sorted_cols = sorted(related_cols.items(), key=lambda item: item[1], reverse=True)
#         top_results[col1] = dict(sorted_cols[:rel_top])
#     return top_results


def extract_functions(code_text: str) -> list[str]:
    # 匹配 def 函数定义及其缩进体（假设函数体是用4个空格缩进）
    pattern = r'def .*?:\n(?: {4}.*\n?)+'
    functions = re.findall(pattern, code_text)
    return functions


def execute_function(function_code: str, val):
    # Define a local scope to execute our function
    local_scope = {}
    exec(function_code, globals(), local_scope)
    function_name = list(local_scope.keys())[0]
    function = local_scope[function_name]
    return function(val)


def generate_rvd_features(dataframe, l_attribute, r_attribute):
    """
    生成Referential Violation Detection (RVD)的特征向量

    参数:
    dataframe: pandas DataFrame对象,包含待检测的数据
    l_attribute: 左列名
    r_attribute: 右列名

    返回:
    feature_vector: 一个numpy数组,表示每个单元格是否违反引用完整性(1表示违反,0表示不违反)
    """
    feature_vector = numpy.zeros(dataframe.shape[0])
    # l_j = dataframe.columns.get_loc(l_attribute)
    # r_j = dataframe.columns.get_loc(r_attribute)

    # 构建值字典
    value_dictionary = {}
    for i, row in dataframe.iterrows():
        if row[l_attribute]:
            if row[l_attribute] not in value_dictionary:
                value_dictionary[row[l_attribute]] = {}
            if row[r_attribute]:
                value_dictionary[row[l_attribute]][row[r_attribute]] = 1

    # 检测违反引用完整性的单元格
    for i, row in dataframe.iterrows():
        if row[l_attribute] in value_dictionary and len(value_dictionary[row[l_attribute]]) > 1:
            feature_vector[i] = 1.0

    return feature_vector


def run_pvd_on_column(dataframe, attribute):
    """
    对指定列执行PVD检测，识别所有字符并返回包含这些字符的位置。

    参数:
    dataframe: pandas DataFrame
    attribute: 要检测的列名（字符串）

    返回:
    outputted_cells: dict，键为(i, j)，表示检测到特殊字符的位置
    configurations: list，包含检测过的字符配置，如 [attribute, ch]
    """
    outputted_cells = {}
    configurations = []

    if attribute not in dataframe.columns:
        raise ValueError(f"列 '{attribute}' 不存在于DataFrame中")

    seen_chars = set()
    for val in dataframe[attribute]:
        if isinstance(val, str):
            seen_chars.update(val)

    j = dataframe.columns.get_loc(attribute)

    for ch in seen_chars:
        configurations.append([attribute, ch])
        pattern = re.compile(f"[{re.escape(ch)}]")
        for i, value in dataframe[attribute].items():
            try:
                if isinstance(value, str) and pattern.search(value):
                    outputted_cells[(i, j)] = ""
            except (TypeError, AttributeError):
                continue

    return outputted_cells, configurations


def compute_nmi_matrix(data):
    data_copy = data.copy()
    nmi_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)

    for col1 in data.columns:
        for col2 in data.columns:
            valid_idx = data_copy[[col1, col2]].dropna().index
            if len(valid_idx) > 0:
                x = data_copy.loc[valid_idx, col1]
                y = data_copy.loc[valid_idx, col2]
                nmi = normalized_mutual_info_score(x, y)
                # nmi_matrix.loc[col1, col2] = round(nmi, 4)
                nmi_matrix.loc[col1, col2] = nmi
            else:
                nmi_matrix.loc[col1, col2] = np.nan  # 无有效数据
            # nmi = normalized_mutual_info_score(df_copy[col1], df_copy[col2])
            # nmi_matrix.loc[col1, col2] = round(nmi, 4)

    return nmi_matrix


def get_top_nmi_relations(data, threshold=0.9, max_top=3):
    nmi_matrix = compute_nmi_matrix(data)
    print(nmi_matrix['State'])
    result = {}

    for col in nmi_matrix.columns:
        scores = nmi_matrix.loc[col].drop(index=col)  # 排除自己
        sorted_scores = scores.sort_values(ascending=False)
        high_nmi = sorted_scores[sorted_scores > threshold]

        if len(high_nmi) > max_top:
            selected = high_nmi.head(max_top)
        elif len(high_nmi) > 0:
            selected = high_nmi
        else:
            selected = sorted_scores.head(1)  # 保底至少一个

        result[col] = list(selected.index)

    return result


def is_clean(label_data, attribute, row:int):
    filtered_df = label_data[(label_data['row_id'] == row) & (label_data['col_name'] == attribute)]
    if filtered_df.empty:
        return None
    if len(filtered_df) > 1:
        raise ValueError("Multiple values found for the same row and column.")

    return filtered_df['is_clean'].iloc[0]


def cal_acc(label_data, attribute, predictions):
    sorted_data = label_data[label_data['col_name'] == attribute].sort_values(by='row_id', ascending=True)
    labels = sorted_data['is_clean'].values
    print(f"total labels: {len(labels)}, total predictions: {len(predictions)}, total clean: {labels.sum()}")

    # print(labels)

    accuracy = accuracy_score(labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def split_list(lst, group_size=100):
    return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
