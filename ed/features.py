from collections import Counter

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import AutoTokenizer

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()


def llm_tokenizer(text: str) -> list[str]:
    """
    Use Llama tokenizer to tokenize the input text.
    :param text: input text to tokenize
    :return: a list of subword tokens
    """
    raw_tokens = llama_tokenizer.tokenize(text)
    final_tokens = []
    for token in raw_tokens:
        if token.isdigit():
            final_tokens.extend(list(token))  # 将数字拆分为字符
        else:
            final_tokens.append(token)
    # print(f"Tokenized {text} to {final_tokens}")
    return final_tokens


def split_character(text: str) -> list[str]:
    """
    Split the input text into individual characters.
    :param text:
    :return:
    """
    if not isinstance(text, str):
        text = str(text)

    return list(text)


def generate_pattern_feature(series: Series, tokenizer=llm_tokenizer, use_tfidf=True, max_features=100):
    """
    Generate pattern features for a given text series. It computes token counts, character lengths, character-based TF-IDF, token binning, and normalizes the features.
    """
    if series is None or series.empty:
        return DataFrame()

    if series.isnull().any():
        print("Warning: Input series contains NaN values. These will be filled with \'np.nan\' strings.")

    texts = series.fillna('').astype(str).tolist()

    # if tokenizer:
    #     # print("using tokenizer:", tokenizer.__name__)
    #     token_counts = [len(tokenizer(text)) if tokenizer(text) else 0 for text in texts]
    # else:
    #     token_counts = [len(text.split()) for text in texts]  # fallback 分词方式

    if use_tfidf:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None
        )
    else:
        vectorizer = CountVectorizer(
            tokenizer=tokenizer,
            token_pattern=None
        )

    binned_matrix, bin_col_names, token_counts = compute_token_bin_features(texts, tokenizer, method='quantile',
                                                                            n_bins=30, normalize=True)

    char_lengths = [len(text) for text in texts]

    token_counts_scaled = minmax_scaler.fit_transform(np.array(token_counts).reshape(-1, 1))
    char_lengths_scaled = minmax_scaler.fit_transform(np.array(char_lengths).reshape(-1, 1))

    char_vectorizer = TfidfVectorizer(tokenizer=split_character, token_pattern=None)

    X_char = char_vectorizer.fit_transform(texts)
    # X_token = vectorizer.fit_transform(texts)
    # if X_token.shape[1] > max_features:
    #     token_matrix = reduce_dimension(X_token, max_features)
    # else:
    #     token_matrix = X_token.toarray()
    # for i in range(len(token_matrix)):
    #     token_matrix[i] = token_matrix[i] / token_counts[i] if token_counts[i] > 0 else token_matrix[i]

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

    n_feats = feature_matrix.shape[1]
    column_names = ([f"feat_{i}" for i in range(n_feats)])

    return DataFrame(feature_matrix, columns=column_names)


def generate_fd_feature(dataframe: DataFrame, rhs_col: str):
    """
    Generate functional dependency features for a DataFrame.
    """
    dataframe = dataframe.astype(str)
    lhs_cols = dataframe.columns

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


def compute_token_bin_features(texts, tokenizer, n_bins=30, method='quantile', normalize=True):
    """
    Calculate token binning features for a list of texts.
    """
    if tokenizer:
        tokenized = [tokenizer(text) if tokenizer(text) else [] for text in texts]
    else:
        tokenized = [text.split() for text in texts]

    flat_tokens = []
    token_cnt = []
    for toks in tokenized:
        token_cnt.append(len(toks))
        for tok in toks:
            flat_tokens.append(tok)

    token_tf = Counter(flat_tokens)
    token_freqs = np.array(list(token_tf.values()))
    n_bins = min(n_bins, len(token_freqs))

    if method == 'quantile':
        bin_edges = np.unique(np.percentile(token_freqs, np.linspace(0, 100, n_bins + 1)))
    elif method == 'uniform':
        bin_edges = np.linspace(token_freqs.min(), token_freqs.max(), n_bins + 1)
    elif method == 'log':
        token_freqs_log = np.log1p(token_freqs)
        bin_edges = np.expm1(np.linspace(token_freqs_log.min(), token_freqs_log.max(), n_bins + 1))
    else:
        raise ValueError("Invalid binning method. Choose from 'quantile', 'uniform', or 'log'.")

    bin_edges = np.unique(bin_edges)  # Ensure uniqueness
    bin_labels = [f"<{b}" for b in bin_edges] + [f">={bin_edges[-1]}"]

    def get_bin_idx(freq):
        return np.searchsorted(bin_edges[1:], freq, side='right')

    # Build feature matrix
    binned_matrix = np.zeros((len(texts), len(bin_labels)))

    for i, toks in enumerate(tokenized):
        for tok in toks:
            units = list(tok) if tok.isdigit() else [tok]
            for unit in units:
                tf = token_tf.get(unit, 0)
                bin_idx = get_bin_idx(tf)
                binned_matrix[i, bin_idx] += 1

    # row normalization
    if normalize:
        row_sums = binned_matrix.sum(axis=1, keepdims=True)
        row_sums += 1e-10  # 避免除以0
        # row_sums[row_sums == 0] = 1  # 避免除以0
        binned_matrix = binned_matrix / row_sums

    return binned_matrix, [f"bin_tf_{label}" for label in bin_labels], token_cnt


def reduce_dimension(feature_vector: ndarray, n_components=None):
    """
    Reduce the dimensionality of the feature vector using Truncated SVD.
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
    """ Generate features for a given DataFrame and target column."""
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


def cluster_features(feature_dataframe: DataFrame, cluster_params, verbose=False):
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
        raise ValueError("Cluster parameters must be provided!")

    # k = cluster_params.get('n_clusters')
    # if verbose:
    #     print("Clustering parameters:", cluster_params)

    # kmeans = KMeans(n_clusters=k, random_state=42)
    # kmeans.fit_predict(feature_dataframe.values)
    # labels = kmeans.labels_
    # centroids = kmeans.cluster_centers_

    # mb_kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=256, max_iter=100, random_state=42)
    mb_kmeans = MiniBatchKMeans(**cluster_params)
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


def propagate_labels(clusters: ndarray, representatives: list | ndarray, rep_labels: list | Series | ndarray):
    """
    Propagate labels based on clustering results.

    :param clusters:
    :param representatives: list of indices of representative samples for each cluster in order of cluster IDs
    :param rep_labels: error labels
    :return: DataFrame with propagated labels
    """
    if len(representatives) != len(rep_labels):
        raise ValueError("Length of representatives must match length of rep_labels.")

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


def cluster_dataset(dataframe: DataFrame, cluster_params=None, verbose=False):
    """
    Cluster the dataset and generate features for each column.
    :param dataframe:
    :param cluster_params:
    :param verbose:
    :return:
    """
    if dataframe.empty:
        raise ValueError("Empty dataframe provided for clustering.")

    all_features = {}
    all_representatives = {}
    all_clusters = {}

    for idx, column in enumerate(dataframe.columns):
        if verbose:
            print(f"\nProcessing column: {column}, {idx + 1} / {len(dataframe.columns)}")

        # 生成特征
        feature_df = generate_features(dataframe, column, tokenizer=llm_tokenizer, use_tfidf=True)
        all_features[column] = feature_df
        if verbose:
            print(
                f"Generated features for column '{column}': {feature_df.shape[0]} rows, {feature_df.shape[1]} features")

        col_cluster_params = cluster_params or {'n_clusters': 20}
        clusters, representatives = cluster_features(feature_df, cluster_params=col_cluster_params, verbose=verbose)
        all_representatives[column] = representatives
        all_clusters[column] = clusters
        if verbose:
            print(
                f"Clustered column '{column}': {len(representatives)} representatives, {np.max(clusters) + 1} clusters")

    cluster_df = pd.DataFrame(all_clusters, index=dataframe.index, columns=dataframe.columns)
    repre_df = pd.DataFrame(all_representatives, columns=dataframe.columns)

    return cluster_df, repre_df


def propagate_labels_from_clusters(
        cluster_dataframe: DataFrame,
        representatives_dataframe: DataFrame,
        error_labels: DataFrame
):
    """
    Propagate error labels based on clustering results.
    """
    if cluster_dataframe.empty or representatives_dataframe.empty:
        raise ValueError("Empty cluster or representatives dataframe provided")

    pred_dataframe = DataFrame(np.zeros_like(error_labels, dtype=bool), columns=error_labels.columns)

    for column in cluster_dataframe.columns:
        clusters = cluster_dataframe[column].values
        representatives = representatives_dataframe[column].values

        valid_representatives = [rep for rep in representatives if rep != -1]

        true_error_mask = error_labels[column].astype(bool)
        selected_labels = true_error_mask[valid_representatives]

        propagated_labels = propagate_labels(clusters, valid_representatives, selected_labels)

        pred_dataframe[column] = propagated_labels

    return pred_dataframe
