import ast
import json
import re

import numpy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, pairwise_distances_argmin_min, \
    precision_score, recall_score, f1_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

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




def split_list(lst, group_size=100):
    return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]


def cluster_feature(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print('center', cluster_centers)
    closest, _ = pairwise_distances_argmin_min(cluster_centers, features)
    print(closest)
    clusters = [np.where(labels == i)[0].tolist() for i in range(n_clusters)]
    print(clusters)
    return clusters, closest


def fix_and_parse_list(bad_list_str):
    items = re.findall(r"""(['"])(.*?)(?<!\\)\1""", bad_list_str)

    fixed_items = []
    for quote, item in items:
        # 去除内部未转义的引号
        clean_item = item.replace('"', '\\"').replace("'", "\\'")
        fixed_items.append(f'"{clean_item}"')

    fixed_list_str = f"[{', '.join(fixed_items)}]"

    try:
        result = ast.literal_eval(fixed_list_str)
        print("fixed")
        return result
    except Exception as e:
        print("解析失败:", e)
        print("修正后的字符串:", fixed_list_str)
        return []


def generate_equality_labels(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    比较两个DataFrame的每一列，支持混合类型（数字和字符串）一致性判断。

    返回一个与原DataFrame结构相同的布尔型DataFrame，标记每个元素是否一致。
    """
    if df1.shape != df2.shape:
        raise ValueError("两个DataFrame的shape不一致")

    label_df = pd.DataFrame(False, index=df1.index, columns=df1.columns)

    for col in df1.columns:
        s1 = df1[col]
        s2 = df2[col]

        if s1.dtype == s2.dtype:
            label_df[col] = s1 == s2
        elif s1.dtype == bool or s2.dtype == bool:
            label_df[col] = s1.map(str) == s2.map(str)
            if col == 'Wealth How Was Political':
                print(s1.map(str))
                print(s2.map(str))

        else:
            s1_num = pd.to_numeric(s1, errors='coerce')
            s2_num = pd.to_numeric(s2, errors='coerce')

            both_numeric = s1_num.notna() & s2_num.notna()
            # label_df[col] = False
            label_df.loc[both_numeric, col] = s1_num[both_numeric] == s2_num[both_numeric]

            # 3. 对非数值部分，统一转为str比较
            non_numeric = ~both_numeric
            s1_str = s1.astype(str)
            s2_str = s2.astype(str)
            label_df.loc[non_numeric, col] = s1_str[non_numeric] == s2_str[non_numeric]

    return label_df


def is_column_numeric(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors='coerce')
    # 只关注非空的行，确保转换后都不是NaN
    return converted[series.notna()].notna().all()


def get_target_related_combination_freq(row, combo_value_cnts, target_column, related_columns, top_k=3):
    related_values = row[related_columns].to_dict()

    # combo_cols = [target_column] + related_columns
    # combo_counts = data[combo_cols].value_counts(normalize=False).reset_index()
    # combo_counts.columns = combo_cols + ['freq']

    combo_counts_target = combo_value_cnts[combo_value_cnts[target_column] == row[target_column]]
    combo_counts_related = combo_value_cnts.copy()

    # 过滤：只保留与当前行的 related 值相同的组合
    for col in related_columns:
        combo_counts_related = combo_value_cnts[combo_value_cnts[col] == related_values[col]]

    # 取前 top_k
    top_combos = pd.concat([combo_counts_related.head(top_k), combo_counts_target.head(top_k)], ignore_index=True)
    top_combos = top_combos.drop_duplicates()
    top_combos = top_combos.sort_values(by='count', ascending=False)
    if top_combos.shape[0] > 1:
        top_combos = top_combos[top_combos['count'] > 1]

    # 转为 list of dict 格式
    records = top_combos.to_dict(orient='records')
    return records


def extract_ans_json(text):
    pattern = r'\{[^{}]*?"is_valid"\s*:\s*[^,{}]+,\s*"reason"\s*:\s*"[^"]*?",\s*"value"\s*:\s*"[^"]*?"\s*\}'

    matches = re.findall(pattern, text, flags=re.DOTALL)
    for match in matches:
        try:
            obj = json.loads(match)
            # 检查结构是否符合要求
            if isinstance(obj, dict) and {"is_valid", "reason", "value"} <= obj.keys():
                return obj
        except json.JSONDecodeError:
            continue

    return None


def extract_json_list(response_text, pattern=None):
    # 正则表达式匹配整个 JSON 数组
    # json_pattern = r'\[\s*(?:\{[^{}]*"input_id"\s*:\s*\d+\s*,\s*"predicted_values"\s*:\s*"[^"]*"[^{}]*\}\s*,?\s*)+\]'
    if pattern is None:
        pattern = r'\[\s*\{.*?\}\s*\]'
    # pattern = r'\[\s*(?:\{\s*"input_id"\s*:\s*\d+\s*,\s*"predicted_values"\s*:\s*\[[^\]]*\]\s*\}\s*,?\s*)+\]'
    response_text = remove_json_comments(response_text)
    match = re.search(pattern, response_text)
    if match:
        json_str = match.group(0)
        try:
            # 解析 JSON 字符串为 Python 对象
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            print(f'Parse error: {json_str}')
            return None
    print("No valid JSON array found in the response.")
    return None


def cluster_columns_by_nmi(nmi_matrix, threshold=0.4)-> dict:
    # 1 - NMI → 距离（越近越相关）
    dist_matrix = 1 - nmi_matrix.values
    np.fill_diagonal(dist_matrix, 0)  # self-distance = 0
    condensed_dist = squareform(dist_matrix, checks=False)

    Z = linkage(condensed_dist, method='average')  # 层次聚类
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')

    column_groups = {}
    for col, label in zip(nmi_matrix.columns, cluster_labels.tolist()):
        column_groups.setdefault(label, []).append(col)

    return column_groups


def get_matching_row_indices(df, related_values_list, target_column):
    result_indices = []
    for related_values in related_values_list:
        condition = pd.Series(True, index=df.index)
        for col, val in related_values.items():
            if col == target_column:
                continue
            if pd.isna(val):
                condition &= df[col].isna()
            else:
                condition &= df[col] == val
        matched_indices = df[condition].index.tolist()
        result_indices.append(matched_indices)
    return result_indices


def label_predictions(label_df, predictions, dataframe, related_values_list, target_column):
    # 根据related_values找到对应的行索引组
    row_groups = get_matching_row_indices(dataframe, related_values_list, target_column)
    print(f'Matched row indexes: {row_groups}')

    for i, prediction in enumerate(predictions):
        pred_value = prediction["predicted_value"]
        row_ids = row_groups[i]

        for row_id in row_ids:
            error_val = dataframe.loc[row_id, target_column]
            is_error = pred_value != error_val
            label_df.loc[row_id, target_column] = is_error


def label_predictions_from_result(label_df, result, dataframe, related_values_list, target_column):
    # 根据related_values找到对应的行索引组
    row_groups = get_matching_row_indices(dataframe, related_values_list, target_column)
    print(f'Matched row indexes: {row_groups}')

    for i, prediction in enumerate(result):
        pred_value = prediction.get("predicted_values")
        # 跳过没预测值的情况
        if pred_value is None:
            continue

        row_ids = row_groups[i]

        for row_id in row_ids:
            error_val = dataframe.loc[row_id, target_column]
            is_error = pred_value != error_val
            label_df.loc[row_id, target_column] = is_error


def label_cells_from_judgments(
        label_df: pd.DataFrame,
        judgments: list,
        data_df: pd.DataFrame,
        combo_frequency: list,
        key_column: str
):
    """
    根据单元格级别错误检测结果，对数据框中的相应单元格进行标注

    参数:
        label_df: 用于标注结果的数据框（与data_df索引对应）
        judgments: 从LLM返回的判断结果，格式为[{"row_id": 1, "column_judgments": {"col1": "REASONABLE", ...}}, ...]
        data_df: 原始数据框
        combo_frequency: 具有相同key_column值的组合列表，格式为[{col1: val1, col2: val2, ..., frequency: freq}, ...]
        key_column: 主键列名

    返回:
        更新后的label_df，其中标记了可疑单元格（值为"SUSPICIOUS"）
    """
    if not judgments:
        return label_df

    # 获取key_column的值（所有组合都有相同的key_column值）
    key_value = combo_frequency[0][key_column]

    # 处理每个判断结果
    for judgment in judgments:
        row_id = judgment.get("row_id")
        column_judgments = judgment.get("column_judgments", {})

        # 找到combo_frequency中对应的行
        if 0 < row_id <= len(combo_frequency):
            combo_row = combo_frequency[row_id - 1]  # row_id是从1开始的
            # 创建查询条件，根据组合中的所有值来匹配行
            query_condition = pd.Series(True, index=data_df.index)
            for col, val in combo_row.items():
                if col == 'frequency':
                    continue
                if pd.isna(val):
                    query_condition &= data_df[col].isna()
                else:
                    query_condition &= data_df[col] == val

            # 找到所有匹配的行索引
            matching_rows = data_df[query_condition].index
            print(f'Total matched rows {len(matching_rows)} for row_id {row_id} with {key_column}: {key_value}, frqequency {combo_row.get("frequency", 0)}')

            # 遍历所有列判断
            for col, status in column_judgments.items():
                for idx in matching_rows:
                    label_df.at[idx, col] = status == "SUSPICIOUS"

    return label_df

def remove_json_comments(json_str):
    # 去除行尾注释 // ...
    no_inline_comments = re.sub(r'//.*', '', json_str)
    return no_inline_comments

def select_key_column(nmi_matrix: pd.DataFrame, column_group: list[str]) -> str:
    """
    在一个列组中选择与其他列最相关的主键列（key column）。
    依据是：对每一列，计算它与组内其他列的 NMI 总和，选择和最大的那一列。
    """
    max_total_nmi = -1
    key_column = None

    for col in column_group:
        sub_nmi = nmi_matrix.loc[col, column_group].drop(col)  # 去掉自身
        total_nmi = sub_nmi.sum()

        if total_nmi > max_total_nmi:
            max_total_nmi = total_nmi
            key_column = col

    return key_column