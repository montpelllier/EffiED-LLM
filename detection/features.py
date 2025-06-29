import concurrent.futures
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from transformers import AutoTokenizer

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
standard_scaler = StandardScaler()


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


def generate_bot_feature(series: Series, tokenizer=None, max_features=None, use_tfidf=True):
    """
    针对单列文本生成 Bag of Tokens 特征

    参数:
        series: pandas Series, 包含文本数据
        tokenizer: 分词函数, 默认None使用TfidfVectorizer默认分词器
        max_features: int, 保留的最大特征数量
        use_tfidf: bool, 是否使用TF-IDF而非CountVectorizer

    返回:
        pd.DataFrame: 特征矩阵
    """
    if series is None or series.empty:
        return DataFrame()

    if series.isnull().any():
        print("Warning: Input series contains NaN values. These will be filled with empty strings.")

    texts = series.fillna('').astype(str).tolist()

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
    feature_matrix = X.toarray()

    if feature_matrix.shape[1] > 500:
        feature_matrix = reduce_dimension(feature_matrix, 500)

    feature_matrix = standard_scaler.fit_transform(feature_matrix)
    # 构造完整输出，空值行补 NaN
    # full_matrix = np.full((len(series), feature_matrix.shape[1]), np.nan)
    # full_matrix[is_valid.values] = feature_matrix

    # 命名列
    feature_names = [f"tok_{tok}" for tok in vectorizer.get_feature_names_out()]
    # return DataFrame(full_matrix, columns=feature_names)
    return DataFrame(feature_matrix)


def generate_fd_feature(dataframe: DataFrame, rhs_col: str):
    """
    构建 FD 特征向量，每行包含来自每个 LHS 对 RHS 的：
    - global support（该 RHS 值在所有 LHS 值中出现的频率）
    - conditional confidence（在该 LHS 值下该 RHS 值的频率）
    """
    if dataframe[rhs_col].isnull().any():
        raise ValueError(f"Column '{rhs_col}' contains NaN values, which are not allowed for FD feature generation.")
    dataframe = dataframe.astype(str)
    lhs_cols = [col for col in dataframe.columns if col != rhs_col]

    # 全局 RHS 值计数
    global_rhs_counter = Counter(dataframe[rhs_col])
    total_rhs = len(dataframe)

    # 构建每列的 LHS → Counter(RHS)
    fd_stats = {}
    for lhs_col in lhs_cols:
        grouped = defaultdict(list)
        for _, row in dataframe.iterrows():
            grouped[row[lhs_col]].append(row[rhs_col])
        fd_stats[lhs_col] = {
            lhs_val: Counter(rhs_vals)
            for lhs_val, rhs_vals in grouped.items()
        }

    # 每行计算特征
    features = []
    for _, row in dataframe.iterrows():
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


def reduce_dimension(feature_vector, n_components=None):
    """
    对 BoT 特征进行标准化和降维。

    参数:
        feature_df: pd.DataFrame，BoT特征（每行一个样本）
        variance_threshold: float，PCA保留的累计解释方差比例

    返回:
        reduced_features: pd.DataFrame，降维后的特征
        pca_model: PCA对象，可用于解释各主成分含义
    """
    scaled_feature = standard_scaler.fit_transform(feature_vector)
    if n_components is None:
        n_components = min(100, int(0.1 * feature_vector.shape[1]))
        # n_components = 500

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(scaled_feature)

    explained = svd.explained_variance_ratio_.sum()
    print(f"information retained by SVD: {explained:.4f}")
    print(f"original dimension: {feature_vector.shape[1]} -> reduced dimension: {reduced.shape[1]}")
    return reduced


def cluster_features(feature_dataframe: DataFrame, method="average", metric="cosine", max_clusters=20):
    """
    Cluster features using hierarchical clustering.

    param
        feature_dataframe: feature vector DataFrame
        method: str, clustering method (e.g., "average", "single", "complete")
        metric: str, distance metric (e.g., "Euclidean", "cosine")
        max_clusters: int, maximum number of clusters to form

    return
        representatives: list, indices of representative samples for each cluster
    """

    if feature_dataframe.empty:
        raise ValueError("Empty feature dataframe provided for clustering.")

    if metric == "cosine" and np.any(np.sum(feature_dataframe.values ** 2, axis=1) == 0):
        print("Warning: Some feature vectors are zero vectors, which may affect cosine distance calculations.")

    dist_matrix = pdist(feature_dataframe.values, metric=metric)
    linkage_matrix = linkage(dist_matrix, method=method)

    clusters = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')
    # clusters = fcluster(linkage_matrix, t=0.6, criterion='distance')
    cluster_num = np.unique(clusters)[-1]
    print(f"Number of clusters formed: {cluster_num}")

    representatives = []
    for cluster_id in range(1, cluster_num + 1):
        indices = np.where(clusters == cluster_id)[0]

        if len(indices) == 0:
            representatives.append(-1)
            continue
        elif len(indices) == 1:
            representatives.append(indices[0])
            continue

        centroid = np.mean(feature_dataframe.iloc[indices].values, axis=0)

        distances = np.array([
            np.linalg.norm(feature_dataframe.iloc[idx].values - centroid)
            for idx in indices
        ])
        closest_point_idx = indices[np.argmin(distances)]
        representatives.append(closest_point_idx)

    return clusters, representatives


def propagate_labels(labels: pd.Series, clusters, representatives):
    """
    Propagate labels based on clustering results.

    :param labels: Series indicating true error labels
    :param clusters:
    :param representatives: list of indices of representative samples for each cluster
    :return: DataFrame with propagated labels
    """

    propagated_labels = np.zeros(len(labels), dtype=bool)

    for cluster_id, rep_idx in enumerate(representatives):
        if rep_idx == -1:
            continue

        label = labels.iloc[rep_idx]
        cluster_indices = np.where(clusters == cluster_id + 1)[0]
        propagated_labels[cluster_indices] = label

    return Series(propagated_labels, index=labels.index)



def cluster_and_propagate(dataframe: DataFrame, error_labels: DataFrame, method="average", metric="cosine",
                          max_clusters=20, feature_params=None, parallel=False, verbose=False):
    """
    为数据集中的每一列聚类特征并传播标签。

    参数:
        dataframe: DataFrame，包含需要处理的数据
        error_labels: DataFrame，包含真实错误标签
        method: str，聚类方法 ("average", "ward", "complete" 等)
        metric: str，距离度量 ("cosine", "euclidean" 等)
        max_clusters: int，最大聚类数
        feature_params: dict，特征生成参数 {'bot': {...}, 'fd': {...}}
        parallel: bool，是否并行处理多个列
        verbose: bool，是否输出详细信息

    返回:
        DataFrame: 每列传播后的预测标签
    """

    # 输入验证
    if dataframe.empty or error_labels.empty:
        raise ValueError("Empty dataframe or error labels provided")

    if not all(col in error_labels.columns for col in dataframe.columns):
        raise ValueError("Dataframe columns must match error labels columns")

    # 默认特征参数
    if feature_params is None:
        feature_params = {
            'bot': {'tokenizer': llm_tokenizer, 'use_tfidf': True},
            'fd': {}
        }

    pred_dataframe = DataFrame(np.zeros_like(error_labels, dtype=bool), columns=error_labels.columns)

    # 处理单个列的函数
    def process_column(column):
        if verbose:
            print(f"Processing column: {column}")

        col_series = dataframe[column]
        missing_mask = col_series.isnull()

        # 1. 生成特征
        try:
            non_missing_index = col_series[~missing_mask].index
            missing_index = col_series[missing_mask].index
            # bot_feature = generate_bot_feature(dataframe[column], **feature_params.get('bot', {}))
            # fd_feature = generate_fd_feature(dataframe, column)
            bot_feature = generate_bot_feature(col_series.loc[non_missing_index], **feature_params.get('bot', {}))
            fd_feature = generate_fd_feature(dataframe.loc[non_missing_index], column)

            feature_df = pd.concat([bot_feature, fd_feature], axis=1)
        except Exception as e:
            print(f"Error generating features for column {column}: {e}")
            return column, Series([False] * len(dataframe), index=dataframe.index)

        # 2. 聚类并传播标签
        try:
            clusters, representatives = cluster_features(feature_df, method=method, metric=metric,
                                                         max_clusters=max_clusters)
            true_error_mask = error_labels[column].astype(bool).loc[non_missing_index]
            # true_error_mask = error_labels[column].astype(bool)
            propagated_labels = propagate_labels(true_error_mask, clusters, representatives)

            full_labels = Series(False, index=dataframe.index)
            full_labels.loc[non_missing_index] = propagated_labels
            full_labels.loc[missing_index] = True

            if verbose:
                error_count = true_error_mask.sum()
                pred_count = full_labels.sum()
                print(f"Column {column}: true errors: {error_count}, predicted errors: {pred_count}")

            return column, full_labels
        except Exception as e:
            print(f"Error clustering and propagating labels for column {column}: {e}")
            return column, Series([False] * len(dataframe), index=dataframe.index)

    # 根据是否并行处理选择执行方式
    if parallel:
        columns_iter = dataframe.columns
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_column, columns_iter))
        for col, labels in results:
            pred_dataframe[col] = labels
    else:
        columns_iter = tqdm(dataframe.columns) if verbose else dataframe.columns
        for col in columns_iter:
            _, labels = process_column(col)
            pred_dataframe[col] = labels
            # print(f"Contains {labels.isnull().sum()} null values in column {col}")
            # print(f"length of column {col}: {len(labels)}, length of dataframe: {len(dataframe)}")
    # total_nulls = pred_dataframe.isnull().sum().sum()
    # print(f"Total null values in prediction dataframe: {total_nulls}")
    return pred_dataframe
