import concurrent.futures
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from kneed import KneeLocator
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import linkage, fcluster, centroid
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from transformers import AutoTokenizer

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()


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


def split_character(text: str):
    if not isinstance(text, str):
        text = str(text)

    return list(text)


def generate_pattern_feature(series: Series, tokenizer=llm_tokenizer, use_tfidf=True, max_features=500):
    """
    针对单列文本生成 Bag of Tokens 特征

    参数:
        series: pandas Series, 包含文本数据
        tokenizer: 分词函数, 默认None使用TfidfVectorizer默认分词器
        use_tfidf: bool, 是否使用TF-IDF而非CountVectorizer

    返回:
        pd.DataFrame: 特征矩阵
    """
    if series is None or series.empty:
        return DataFrame()

    if series.isnull().any():
        print("Warning: Input series contains NaN values. These will be filled with empty strings.")

    texts = series.fillna('').astype(str).tolist()

    if tokenizer:
        # print("using tokenizer:", tokenizer.__name__)
        token_counts = [len(tokenizer(text)) if tokenizer(text) else 0 for text in texts]
    else:
        token_counts = [len(text.split()) for text in texts]  # fallback 分词方式

    char_lengths = [len(text) for text in texts]

    token_counts_scaled = minmax_scaler.fit_transform(np.array(token_counts).reshape(-1, 1))
    char_lengths_scaled = minmax_scaler.fit_transform(np.array(char_lengths).reshape(-1, 1))


    if use_tfidf:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None
        )
        # vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
    else:
        vectorizer = CountVectorizer(
            tokenizer=tokenizer,
            token_pattern=None
        )
    binned_matrix, bin_col_names = compute_token_bin_features(texts, tokenizer)

    char_vectorizer = TfidfVectorizer(tokenizer=split_character, token_pattern=None)

    X_char = char_vectorizer.fit_transform(texts)
    X_token = vectorizer.fit_transform(texts)

    if X_token.shape[1] > max_features:
        token_matrix = reduce_dimension(X_token, max_features)
    else:
        token_matrix = X_token.toarray()

    for i in range(len(token_matrix)):
        token_matrix[i] = token_matrix[i] / token_counts[i] if token_counts[i] > 0 else token_matrix[i]

    # print(token_matrix.shape, X_char.shape, binned_matrix.shape, token_counts_scaled.shape, char_lengths_scaled.shape)
    # token_matrix = X.toarray()
    # token_matrix = np.hstack([X_char.toarray()])
    # token_matrix = np.hstack([binned_matrix])
    # token_matrix = np.hstack([binned_matrix, X_char.toarray()])
    feature_matrix = np.hstack([
        binned_matrix,
        # token_matrix,
        X_char.toarray(),
        token_counts_scaled,
        char_lengths_scaled
    ])

    # feature_matrix = np.hstack([token_matrix, token_counts_scaled, char_lengths_scaled])

    n_feats = feature_matrix.shape[1]
    column_names = ([f"feat_{i}" for i in range(n_feats)])

    return DataFrame(feature_matrix, columns=column_names)


def generate_fd_feature(dataframe: DataFrame, rhs_col: str):
    """
    构建 FD 特征向量，每行包含来自每个 LHS 对 RHS 的：
    - global support（该 RHS 值在所有 LHS 值中出现的频率）
    - conditional confidence（在该 LHS 值下该 RHS 值的频率）
    """
    if dataframe[rhs_col].isnull().any():
        print(f"Column '{rhs_col}' contains NaN values, which are not allowed for FD feature generation.")

    dataframe = dataframe.astype(str)
    # lhs_cols = [col for col in dataframe.columns if col != rhs_col]
    lhs_cols =dataframe.columns

    total_rows = len(dataframe)

    # 计算 LHS-RHS 组合的全局计数
    lhs_rhs_counts = {}
    lhs_counts = {}
    for lhs_col in lhs_cols:
        lhs_rhs_counts[lhs_col] = Counter()
        lhs_counts[lhs_col] = Counter()

    for _, row in dataframe.iterrows():
        rhs_val = row[rhs_col]
        for lhs_col in lhs_cols:
            lhs_val = row[lhs_col]
            lhs_rhs_counts[lhs_col][(lhs_val, rhs_val)] += 1
            lhs_counts[lhs_col][lhs_val] += 1

    features = []
    for _, row in dataframe.iterrows():
        row_features = {}
        rhs_val = row[rhs_col]

        for lhs_col in lhs_cols:
            lhs_val = row[lhs_col]

            combo_count = lhs_rhs_counts[lhs_col].get((lhs_val, rhs_val), 0)
            lhs_count = lhs_counts[lhs_col].get(lhs_val, 0)

            support = combo_count / total_rows if total_rows else 0.0
            confidence = combo_count / lhs_count if lhs_count else 0.0

            row_features[f"{lhs_col}_support"] = support
            row_features[f"{lhs_col}_confidence"] = confidence

        features.append(row_features)

    feature_df = pd.DataFrame(features)
    feature_df = feature_df / (len(dataframe.columns) ** 0.5)  # 平均化特征值

    return feature_df


def generate_semantic_features(
    series: Series,
    normalize: bool = True,
    batch_size: int = 32
) -> DataFrame:
    """
    为文本列生成语义向量（使用 SentenceTransformer）

    参数:
        series: pd.Series，包含文本数据
        model_name: str，SentenceTransformer 模型名（如 all-MiniLM-L6-v2）
        normalize: 是否对向量做单位归一化（即余弦距离可用）
        batch_size: 批处理大小（适合大数据量）

    返回:
        pd.DataFrame：每行是原始文本的 embedding 向量
    """
    if series is None or series.empty:
        return DataFrame()


    # 填补缺失，转为字符串
    texts = series.fillna("").astype(str).tolist()

    # 编码为语义向量
    embeddings = embed_model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)

    # 转换为 DataFrame
    dim = embeddings.shape[1]
    column_names = [f"sem_{i}" for i in range(dim)]

    return pd.DataFrame(embeddings, columns=column_names)


def compute_token_bin_features(texts, tokenizer, bin_edges=None, normalize=True):
    """
    将表格列中的文本转换为基于 token 全列频率的分箱向量。
    对纯数字 token 按字符拆分统计。
    最后对每行向量进行正则化（行归一化）。

    Args:
        texts (List[str]): 文本数据列表
        tokenizer: 分词函数
        bin_edges (List[int], optional): 分箱右边界（不含），最后一箱为最大值以上
        normalize (bool): 是否对每行特征向量进行归一化

    Returns:
        np.ndarray: binned_matrix [n_samples, n_bins]
        List[str]: bin_col_names
    """

    if bin_edges is None:
        bin_edges = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000]

    if tokenizer:
        tokenized = [tokenizer(text) if tokenizer(text) else [] for text in texts]
    else:
        tokenized = [text.split() for text in texts]
    # 统计词频时数字token拆成字符
    flat_tokens = []
    for toks in tokenized:
        for tok in toks:
            if tok.isdigit():
                flat_tokens.extend(list(tok))
            else:
                flat_tokens.append(tok)

    token_tf = Counter(flat_tokens)

    bin_labels = [f"≤{b}" for b in bin_edges] + [f">{bin_edges[-1]}"]
    bin_index = {label: idx for idx, label in enumerate(bin_labels)}

    def assign_bin(freq):
        for b in bin_edges:
            if freq <= b:
                return f"≤{b}"
        return f">{bin_edges[-1]}"

    binned_matrix = np.zeros((len(texts), len(bin_labels)))

    # 这里也拆数字token，保证统计一致性
    for i, toks in enumerate(tokenized):
        for tok in toks:
            units = list(tok) if tok.isdigit() else [tok]
            for unit in units:
                tf = token_tf.get(unit, 0)
                bin_label = assign_bin(tf)
                binned_matrix[i, bin_index[bin_label]] += 1

    # 行归一化，避免长度差异影响
    if normalize:
        row_sums = binned_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        binned_matrix = binned_matrix / row_sums

    return binned_matrix, [f"bin_tf_{label}" for label in bin_labels]


def reduce_dimension(feature_vector: ndarray, n_components=None):
    """
    对 BoT 特征进行标准化和降维。

    参数:
        feature_df: pd.DataFrame，BoT特征（每行一个样本）
        variance_threshold: float，PCA保留的累计解释方差比例

    返回:
        reduced_features: pd.DataFrame，降维后的特征
    """
    # scaled_feature = standard_scaler.fit_transform(feature_vector)
    scaled_feature = feature_vector

    if n_components is None:
        n_components = min(100, int(0.1 * feature_vector.shape[1]))
        # n_components = 500

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(scaled_feature)

    explained = svd.explained_variance_ratio_.sum()
    print(f"information retained by SVD: {explained:.4f}")
    print(f"original dimension: {feature_vector.shape[1]} -> reduced dimension: {reduced.shape[1]}")
    return reduced


def generate_features(dataframe: DataFrame, target_column: str, tokenizer=None, use_tfidf=None):
    kwargs = {}
    if tokenizer is not None:
        kwargs['tokenizer'] = tokenizer
    if use_tfidf is not None:
        kwargs['use_tfidf'] = use_tfidf

    bot_feature = generate_pattern_feature(dataframe[target_column], **kwargs)
    fd_feature = generate_fd_feature(dataframe, target_column)
    # embed_feature = generate_semantic_features(dataframe[target_column])

    # feature_dataframe = pd.concat([bot_feature, fd_feature, embed_feature], axis=1)
    feature_dataframe = pd.concat([bot_feature, fd_feature], axis=1)
    scaled_features = standard_scaler.fit_transform(feature_dataframe)
    feature_dataframe = pd.DataFrame(scaled_features, columns=feature_dataframe.columns)

    return feature_dataframe


def cluster_features(feature_dataframe: DataFrame, cluster_params=None, verbose=False):
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

    if cluster_params is None:
        raise ValueError("cluster_params must be provided with 't' and 'criterion' keys.")

    k = cluster_params.get('n_clusters')
    if verbose:
        print("Clustering parameters:", cluster_params)

    # kmeans = KMeans(n_clusters=k, random_state=42)
    # kmeans.fit_predict(feature_dataframe.values)
    # labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    mb_kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=256, max_iter=100, random_state=42)
    mb_kmeans.fit_predict(feature_dataframe.values)
    labels = mb_kmeans.labels_
    centroids = mb_kmeans.cluster_centers_

    cluster_num = np.max(labels) + 1
    if verbose:
        print(f"Number of clusters formed: {cluster_num}")

    representatives = []
    for cluster_id in range(cluster_num):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = feature_dataframe.iloc[cluster_indices]

        if len(cluster_indices) == 0:
            representatives.append(-1)
            print(f"Warning: Cluster {cluster_id} has no members.")
            continue
        elif len(cluster_indices) == 1:
            representatives.append(cluster_indices[0])
            # print(f"Cluster {cluster_id} has only one member, using it as representative.")
            continue

        distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
        closest_idx_in_cluster = np.argmin(distances)
        representatives.append(cluster_indices[closest_idx_in_cluster])
    # print(representatives)
    return labels, representatives


def check_clusters(series, labels, clusters, representatives: list):
    for idx, rep_idx in enumerate(representatives):
        if rep_idx == -1:
            continue

        cluster_id = idx + 1
        cluster_indices = np.where(clusters == cluster_id)[0]

        if len(cluster_indices) == 0:
            print(f"Cluster {cluster_id} has no members.")
            continue

        print(f"Cluster {cluster_id} representative index: {rep_idx}, members: {len(cluster_indices)}")
        print("Representative sample:", series.iloc[rep_idx])
        print("Cluster members:")
        members = series.iloc[cluster_indices]
        unique_values = members.unique()
        print(f"Unique values in cluster {cluster_id}: {unique_values}")
        err_num = labels[cluster_indices].sum()
        print(f"Number of errors in cluster {cluster_id}: {err_num}")


def detect_outliers(feature_vectors: ndarray, k=5):
    print(k)
    # 1. 选择 MinPts 值 (例如 4)
    min_pts = k

    # 2. 计算每个点到其第 MinPts 个最近邻居的距离
    # n_neighbors 设置为 MinPts + 1 是因为 NearestNeighbors 会包含点本身
    neigh = NearestNeighbors(n_neighbors=min_pts + 1)
    neigh.fit(feature_vectors)
    distances, indices = neigh.kneighbors(feature_vectors)
    # 3. 获取到第 MinPts 个最近邻居的距离并排序
    k_distances = np.sort(distances[:, min_pts], axis=0)

    # 4. 使用 KneeLocator 找到肘点
    # S=1.0 是敏感度参数，越小越敏感，越大越能找到更明显的肘点
    # curve="convex" 和 direction="increasing" 适用于 K-距离图的形状
    kneedle = KneeLocator(
        x=range(len(k_distances)),
        y=k_distances,
        S=1.0,  # 推荐尝试不同的S值，例如0.5, 1.0, 1.5, 2.0
        curve="convex",
        direction="increasing"
    )

    # 获取肘点的索引和对应的 epsilon 值
    elbow_index = kneedle.knee
    elbow_epsilon = kneedle.elbow_y

    print(f"自动计算的肘点索引 (在排序数组中): {elbow_index}")
    print(f"建议的 epsilon 值: {elbow_epsilon:.4f}")

    elbow_epsilon = min(0.3, elbow_epsilon)  # 限制 epsilon 的最大值为 0.3

    dbscan = DBSCAN(eps=elbow_epsilon, min_samples=min_pts)
    clusters = dbscan.fit_predict(feature_vectors)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"\n使用建议的 epsilon={elbow_epsilon:.4f} 和 MinPts={min_pts}，DBSCAN 发现的簇数: {n_clusters}")
    # print(f"聚类结果: {clusters}")
    outliers = np.where(clusters == -1)[0]

    return outliers


def propagate_labels(clusters: ndarray, representatives: list, rep_labels: list | Series | ndarray):
    """
    Propagate labels based on clustering results.

    :param clusters:
    :param representatives: list of indices of representative samples for each cluster in order of cluster IDs
    :param rep_labels: error labels
    :return: DataFrame with propagated labels
    """
    if len(representatives) != len(rep_labels):
        raise ValueError("Length of representatives must match length of rep_labels.")

    if len(representatives) != np.max(clusters) + 1:
        raise ValueError("Length of representatives must match number of clusters.")

    # print(f"Number of clusters: {np.max(clusters) + 1}, Number of representatives: {len(representatives)}, Number of labels: {len(rep_labels)}")

    propagated_labels = np.zeros(len(clusters), dtype=bool)

    for idx, rep_idx in enumerate(representatives):
        if rep_idx == -1:
            continue

        if isinstance(rep_labels, Series):
            label = rep_labels.iloc[idx]
        elif isinstance(rep_labels, ndarray) or isinstance(rep_labels, list):
            label = rep_labels[idx]
        else:
            raise ValueError("rep_labels must be a Series, ndarray, or list.")

        cluster_indices = np.where(clusters == idx)
        propagated_labels[cluster_indices] = label

    return propagated_labels


def cluster_and_propagate(dataframe: DataFrame, error_labels: DataFrame, cluster_params=None, parallel=False, verbose=False):
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

    pred_dataframe = DataFrame(np.zeros_like(error_labels, dtype=bool), columns=error_labels.columns)

    # 处理单个列的函数
    def process_column(column, col_cluster_params=None):
        nonlocal cluster_params

        if verbose:
            print(f"\nProcessing column: {column}")

        feature_df = generate_features(dataframe, column, tokenizer=llm_tokenizer, use_tfidf=True)
        unique_cnts = dataframe[column].nunique()
        print(f"Unique values in column '{column}': {unique_cnts}")

        if col_cluster_params is None:
            col_cluster_params = {
                'n_clusters': 30,
            }

        try:

            clusters, representatives = cluster_features(feature_df, cluster_params=col_cluster_params)

            # if column in ['city']:
            #     check_clusters(dataframe[column], error_labels[column], clusters, representatives)

            true_error_mask = error_labels[column].astype(bool)
            selected_labels = true_error_mask[representatives]
            propagated_labels = propagate_labels(clusters, representatives, selected_labels)

            full_labels = propagated_labels
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
            _, labels = process_column(col, cluster_params)
            pred_dataframe[col] = labels

    return pred_dataframe


def cluster_dataset(dataframe: DataFrame, cluster_params=None, verbose=False):
    """
    为整个数据集生成特征，进行聚类并传播标签

    参数:
        dataframe: DataFrame，包含需要处理的数据
        error_labels: DataFrame，包含真实错误标签
        cluster_params: dict，聚类参数
        verbose: bool，是否输出详细信息

    返回:
        tuple: (预测标签DataFrame, 特征DataFrame, 代表点DataFrame)
    """
    if dataframe.empty:
        raise ValueError("空的数据框或错误标签")

    # 初始化结果
    # pred_dataframe = DataFrame(np.zeros_like(error_labels, dtype=bool), columns=error_labels.columns)
    all_features = {}
    all_representatives = {}
    all_clusters = {}

    # 处理每一列
    for column in dataframe.columns:
        if verbose:
            print(f"\nProcessing column: {column}")

        # 生成特征
        feature_df = generate_features(dataframe, column, tokenizer=llm_tokenizer, use_tfidf=True)
        all_features[column] = feature_df

        # 聚类
        col_cluster_params = cluster_params or {'n_clusters': 30}

        clusters, representatives = cluster_features(feature_df, cluster_params=col_cluster_params)
        all_representatives[column] = representatives
        all_clusters[column] = clusters

    cluster_df = pd.DataFrame(all_clusters, index=dataframe.index, columns=dataframe.columns)
    repre_df = pd.DataFrame(all_representatives, columns=dataframe.columns)

    return cluster_df, repre_df


def propagate_labels_from_clusters(
    cluster_dataframe: DataFrame,
    representatives_dataframe: DataFrame,
    error_labels: DataFrame,
    verbose=False
):
    """
    从聚类结果中传播标签。

    参数:
        feature_dataframe: DataFrame，特征矩阵
        cluster_dataframe: DataFrame，聚类结果
        representatives_dataframe: DataFrame，代表点索引
        error_labels: DataFrame，真实错误标签
        verbose: bool，是否输出详细信息

    返回:
        DataFrame: 每列传播后的预测标签
    """
    if cluster_dataframe.empty or representatives_dataframe.empty:
        raise ValueError("Empty cluster or representatives dataframe provided")

    pred_dataframe = DataFrame(np.zeros_like(error_labels, dtype=bool), columns=error_labels.columns)

    for column in cluster_dataframe.columns:
        if verbose:
            print(f"\nProcessing column: {column}")

        clusters = cluster_dataframe[column].values
        representatives = representatives_dataframe[column].values

        true_error_mask = error_labels[column].astype(bool)
        selected_labels = true_error_mask[representatives]

        propagated_labels = propagate_labels(clusters, representatives, selected_labels)

        pred_dataframe[column] = propagated_labels

    return pred_dataframe