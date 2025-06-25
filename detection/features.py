import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import AutoTokenizer

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")


def llm_tokenizer(text):
    """
    Use Llama tokenizer to tokenize the input text.
    :param text: input text to tokenize
    :return: a list of subword tokens
    """
    raw_tokens = llama_tokenizer.tokenize(text)
    final_tokens = []
    for token in raw_tokens:
        if token.isdigit():
            final_tokens.extend(['digit'] * len(token))  # 将数字替换为 'digit' token)
        else:
            final_tokens.append(token)
    # print(f"Tokenized {text} to {final_tokens}")
    return final_tokens


def generate_bot_feature(series, tokenizer=None, max_features=None, use_tfidf=True,
                         normalize=True, norm_method='standard'):
    """
    针对单列文本生成 Bag of Tokens 特征

    参数:
        series: pandas Series, 包含文本数据
        tokenizer: 分词函数, 默认None使用TfidfVectorizer默认分词器
        max_features: int, 保留的最大特征数量
        use_tfidf: bool, 是否使用TF-IDF而非CountVectorizer
        normalize: bool, 是否标准化特征
        norm_method: str, 标准化方法 ('standard' 或 'minmax')

    返回:
        pd.DataFrame: 特征矩阵
    """
    # 转换输入为字符串列表
    texts = series.astype(str).tolist()

    # 根据参数选择向量化器
    if use_tfidf:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            max_features=max_features,
            token_pattern=None
        )
    else:
        vectorizer = CountVectorizer(
            tokenizer=tokenizer,
            max_features=max_features,
            token_pattern=None
        )

    # 生成特征矩阵
    X = vectorizer.fit_transform(texts)
    feature_names = [f"tok_{tok}" for tok in vectorizer.get_feature_names_out()]

    # 转换为稠密矩阵
    feature_matrix = X.toarray()

    # 根据需要进行标准化
    if normalize:
        if norm_method == 'standard':
            scaler = StandardScaler()
        else:  # minmax
            scaler = MinMaxScaler()

        feature_matrix = scaler.fit_transform(feature_matrix)

    # 转换为DataFrame并返回
    return pd.DataFrame(feature_matrix, columns=feature_names)


def generate_fd_feature(df: pd.DataFrame, rhs_col: str):
    """
    构建 FD 特征向量，每行包含来自每个 LHS 对 RHS 的：
    - global support（该 RHS 值在所有 LHS 值中出现的频率）
    - conditional confidence（在该 LHS 值下该 RHS 值的频率）
    """
    df = df.astype(str)
    lhs_cols = [col for col in df.columns if col != rhs_col]

    # 全局 RHS 值计数
    global_rhs_counter = Counter(df[rhs_col])
    total_rhs = len(df)

    # 构建每列的 LHS → Counter(RHS)
    fd_stats = {}
    for lhs_col in lhs_cols:
        grouped = defaultdict(list)
        for _, row in df.iterrows():
            grouped[row[lhs_col]].append(row[rhs_col])
        fd_stats[lhs_col] = {
            lhs_val: Counter(rhs_vals)
            for lhs_val, rhs_vals in grouped.items()
        }

    # 每行计算特征
    features = []
    for _, row in df.iterrows():
        row_features = {}
        rhs_val = row[rhs_col]

        for lhs_col in lhs_cols:
            lhs_val = row[lhs_col]
            rhs_counter = fd_stats[lhs_col].get(lhs_val, Counter())

            conditional_total = sum(rhs_counter.values())
            conditional_count = rhs_counter.get(rhs_val, 0)
            confidence = conditional_count / conditional_total if conditional_total else 0.0

            global_count = global_rhs_counter.get(rhs_val, 0)
            support = global_count / total_rhs if total_rhs else 0.0

            row_features[f"{lhs_col}_support"] = support
            row_features[f"{lhs_col}_confidence"] = confidence
            # row_features[f"{lhs_col}_quality"] = support * confidence

        features.append(row_features)

    # 转为 DataFrame
    feature_df = pd.DataFrame(features)
    # print(features)

    # 标准化
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_df.columns)

    return scaled_df



