"""
Feature Extractor
Extract data features for error detection, based on ed/features.py implementation
"""
import collections
import warnings
from typing import List, Tuple

import numpy
import pandas
import scipy
import sklearn
import transformers


class FeatureExtractor:
    """
    A feature extractor for data error detection.

    This class extracts various types of features from text data including:
    - Basic statistical features (character counts, ratios, etc.)
    - Character-level TF-IDF features
    - Token binning features using LLM tokenizer
    - Functional dependency features for rule-based detection

    Attributes:
        llm_tokenizer: Pre-trained LLM tokenizer for text processing
        standard_scaler: StandardScaler for feature normalization
        minmax_scaler: MinMaxScaler for feature scaling
    """

    def __init__(self, tokenizer_model: str = "meta-llama/Llama-3.2-3B"):
        """
        Initialize the FeatureExtractor with specified tokenizer model.

        Args:
            tokenizer_model: Name of the pre-trained tokenizer model to use
        """
        # Initialize tokenizer and embedding models
        self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model)
        self.standard_scaler = sklearn.preprocessing.StandardScaler()
        self.minmax_scaler = sklearn.preprocessing.MinMaxScaler()

    @staticmethod
    def tokenize(llm_tokenizer, text: str) -> List[str]:
        """
        Use LLM tokenizer to tokenize input text. Number tokens are split into individual characters.

        Args:
            llm_tokenizer: The tokenizer instance to use
            text: Input text to tokenize

        Returns:
            List of tokens with numbers split into individual characters
        """
        raw_tokens = llm_tokenizer.tokenize(text)
        final_tokens = []
        for token in raw_tokens:
            if token.isdigit():
                final_tokens.extend(list(token))  # Split numbers into characters
            else:
                final_tokens.append(token)
        return final_tokens

    def extract_pattern_feature(self, texts: List[str], char_config=None, tkn_config=None) -> pandas.DataFrame:
        """
        Extract comprehensive pattern features from text data.

        This method combines multiple feature types:
        - Basic statistical features (character counts, ratios)
        - Character-level TF-IDF features
        - Token binning features with frequency analysis

        Args:
            texts: List of text strings to extract features from
            char_config: Configuration dict for character-level features
            tkn_config: Configuration dict for token-level features

        Returns:
            DataFrame containing all extracted pattern features
        """
        if not texts:
            return pandas.DataFrame()

        # Extract basic features
        basic_features, basic_feat_names = self._extract_basic_features(texts)

        # Extract character-level features
        if char_config is None:
            char_config = {
                'analyzer': 'char',
                'max_features': 200,
            }
        char_feature, char_feat_names = self._extract_character_feature(texts, char_config)

        # Extract token-level features
        if tkn_config is None:
            tkn_config = {
                'n_bins': 50,
                'method': 'quantile',
                'normalize': True
            }
        binned_matrix, bin_feat_names, token_counts = self._bin_tokens(texts, **tkn_config)
        token_cnts_scaled = self.minmax_scaler.fit_transform(
            numpy.array(token_counts).reshape(-1, 1)
        )
        token_feature = numpy.column_stack([binned_matrix, token_cnts_scaled])
        token_feat_names = bin_feat_names + ['token_count_scaled']

        pattern_feature = self._merge_features(
            [basic_features, char_feature, token_feature],
            [basic_feat_names, char_feat_names, token_feat_names]
        )

        return pattern_feature

    @staticmethod
    def extract_rule_feature(dataframe: pandas.DataFrame, rhs_col: str) -> pandas.DataFrame:
        """
        Generate functional dependency features (support and confidence) for DataFrame.

        This method computes functional dependency relationships between columns,
        calculating support and confidence metrics for each left-hand side column
        with respect to the right-hand side column.

        Args:
            dataframe: Input DataFrame to analyze
            rhs_col: Right-hand side column name (dependent variable)

        Returns:
            DataFrame with support and confidence features for each column pair
        """
        df = dataframe.astype(str)
        total_rows = len(df)
        lhs_cols = df.columns

        feature_frames = []

        for lhs_col in lhs_cols:
            # Skip self-dependency (rhs_col -> rhs_col)
            if lhs_col == rhs_col:
                # Create special features for self-dependency
                feature_frame = pandas.DataFrame({
                    f"{lhs_col}_support": [1.0] * total_rows,  # Self support is always 1
                    f"{lhs_col}_confidence": [1.0] * total_rows  # Self confidence is always 1
                })
                feature_frames.append(feature_frame)
                continue

            # Calculate lhs, rhs combination counts
            lhs_rhs_count = df.groupby([lhs_col, rhs_col]).size().reset_index(name="combo_count")
            # Calculate lhs counts
            lhs_count = df.groupby(lhs_col).size().reset_index(name="lhs_count")

            # Merge counts
            merged = df[[lhs_col, rhs_col]].merge(lhs_rhs_count, on=[lhs_col, rhs_col], how="left")
            merged = merged.merge(lhs_count, on=lhs_col, how="left")

            # Calculate support and confidence
            merged[f"{lhs_col}_support"] = merged["combo_count"] / total_rows
            merged[f"{lhs_col}_confidence"] = merged["combo_count"] / merged["lhs_count"]

            # Keep only feature columns
            feature_frames.append(merged[[f"{lhs_col}_support", f"{lhs_col}_confidence"]])

        # Concatenate all features
        feature_df = pandas.concat(feature_frames, axis=1)

        # Normalize feature values
        feature_df /= (len(lhs_cols) ** 0.5)

        return feature_df

    @staticmethod
    def _extract_basic_features(texts: List[str]) -> Tuple[numpy.ndarray, List[str]]:
        """
        Extract basic statistical features from text data.

        Features include character counts, digit/alpha ratios, punctuation counts, etc.

        Args:
            texts: List of text strings

        Returns:
            Tuple of (feature matrix, feature names)
        """
        features = []
        feature_names = [
            'char_count', 'digit_count', 'alpha_count', 'space_count',
            'punct_count', 'upper_count', 'lower_count',
            'digit_ratio', 'alpha_ratio', 'upper_ratio', 'punct_ratio'
        ]

        for text in texts:
            text_len = len(text)

            char_count = text_len
            digit_count = sum(c.isdigit() for c in text)
            alpha_count = sum(c.isalpha() for c in text)
            space_count = text.count(' ')
            punct_count = sum(not c.isalnum() and not c.isspace() for c in text)
            upper_count = sum(c.isupper() for c in text)
            lower_count = sum(c.islower() for c in text)

            if text_len > 0:
                digit_ratio = digit_count / text_len
                alpha_ratio = alpha_count / text_len
                upper_ratio = upper_count / text_len
                punct_ratio = punct_count / text_len
            else:
                digit_ratio = alpha_ratio = upper_ratio = punct_ratio = 0.0

            features.append([
                char_count, digit_count, alpha_count, space_count,
                punct_count, upper_count, lower_count,
                digit_ratio, alpha_ratio, upper_ratio, punct_ratio
            ])

        return numpy.array(features), feature_names

    @staticmethod
    def _extract_character_feature(texts: List[str], config) -> Tuple[numpy.ndarray, List[str]]:
        """
        Extract character-level TF-IDF features.

        Args:
            texts: List of text strings
            config: Configuration dictionary for TfidfVectorizer

        Returns:
            Tuple of (TF-IDF feature matrix, feature names)
        """
        char_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(**config)
        tfidf_matrix = char_vectorizer.fit_transform(texts)
        feature_names = [f"char_{name}" for name in char_vectorizer.get_feature_names_out()]
        return tfidf_matrix.toarray(), feature_names

    def _bin_tokens(self, texts: List[str], n_bins: int = 50, method: str = 'quantile',
                    normalize: bool = True) -> Tuple[numpy.ndarray, List[str], List[int]]:
        """
        Compute token binning features from text data (optimized version).

        This method tokenizes text, calculates token frequency distributions,
        and creates binned features based on token frequencies.

        Args:
            texts: List of text strings
            n_bins: Number of bins for token frequency binning
            method: Binning method ('quantile', 'uniform', 'log')
            normalize: Whether to normalize the binned features

        Returns:
            Tuple of (binned feature matrix, bin column names, token counts)
        """
        if not texts:
            return numpy.array([]), [], []

        # Batch tokenize and preprocess
        tokenized = [self.tokenize(self.llm_tokenizer, text) for text in texts]
        token_counts = [len(tokens) for tokens in tokenized]

        # Build token frequency statistics efficiently
        all_tokens = []
        token_to_text_map = []  # Record which text each token belongs to

        for text_idx, tokens in enumerate(tokenized):
            for token in tokens:
                all_tokens.append(token)
                token_to_text_map.append(text_idx)

        # Calculate global token frequency
        token_counter = collections.Counter(all_tokens)
        unique_tokens = list(token_counter.keys())
        token_freqs = numpy.array(list(token_counter.values()))

        if len(token_freqs) == 0:
            return numpy.zeros((len(texts), 1)), ["bin_tf_empty"], token_counts

        # Optimize binning calculation
        n_bins = min(n_bins, len(numpy.unique(token_freqs)))

        if method == 'quantile':
            # Use more efficient percentile calculation
            percentiles = numpy.linspace(0, 100, n_bins)
            bin_edges = numpy.percentile(token_freqs, percentiles)
        elif method == 'uniform':
            bin_edges = numpy.linspace(token_freqs.min(), token_freqs.max(), n_bins)
        elif method == 'log':
            # Avoid log(0) issue
            token_freqs_safe = numpy.maximum(token_freqs, 1)
            log_freqs = numpy.log(token_freqs_safe)
            log_edges = numpy.linspace(log_freqs.min(), log_freqs.max(), n_bins)
            bin_edges = numpy.exp(log_edges)
        else:
            raise ValueError(f"Invalid binning method '{method}'. Choose from 'quantile', 'uniform', or 'log'.")

        # Ensure bin_edges are unique and sorted
        bin_edges = numpy.unique(bin_edges)

        # Batch calculate bin indices
        token_bin_indices = numpy.digitize(token_freqs, bin_edges) - 1

        # Build token to bin mapping
        token_to_bin = dict(zip(unique_tokens, token_bin_indices))

        # Efficiently build feature matrix
        row_indices = []
        col_indices = []
        data = []

        for token, text_idx in zip(all_tokens, token_to_text_map):
            bin_idx = token_to_bin[token]
            row_indices.append(text_idx)
            col_indices.append(bin_idx)
            data.append(1)

        # Build sparse matrix and convert to dense
        sparse_matrix = scipy.sparse.csr_matrix((data, (row_indices, col_indices)),
                                                shape=(len(texts), n_bins))
        binned_matrix = sparse_matrix.toarray()

        # Normalization
        if normalize:
            row_sums = binned_matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            binned_matrix = binned_matrix / row_sums

        # Generate column names
        bin_labels = [f"bin_tf_{i + 1}_[{bin_edges[i]},{bin_edges[i + 1]})" for i in range(n_bins - 1)]
        bin_labels.append(f"bin_tf_{n_bins}_[{bin_edges[-1]},+inf)")  # Last column

        return binned_matrix, bin_labels, token_counts

    @staticmethod
    def _merge_features(features: List[numpy.ndarray], feature_names: List[List[str]]) -> pandas.DataFrame:
        """
        Merge multiple feature matrices into a single DataFrame.

        Args:
            features: List of feature matrices to merge
            feature_names: List of feature name lists corresponding to each matrix

        Returns:
            Combined DataFrame with all features
        """
        if not features:
            return pandas.DataFrame()

        feature_matrix = numpy.hstack(features)

        if not feature_names:
            feature_names = [['feat_{}'.format(i) for i in range(feature_matrix.shape[1])]]
        column_names = [name for names in feature_names for name in names]

        return pandas.DataFrame(feature_matrix, columns=column_names)

    def extract_column_feature(self, dataframe: pandas.DataFrame, target_column: str) -> pandas.DataFrame:
        """
        Generate comprehensive features for a given DataFrame and target column.

        This method combines pattern features and functional dependency features,
        then applies standard scaling to normalize the final feature set.

        Args:
            dataframe: Input DataFrame to analyze
            target_column: Name of the target column for feature extraction

        Returns:
            DataFrame containing normalized comprehensive features

        Raises:
            ValueError: If target column not found or DataFrame is empty
        """
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
        if dataframe.empty:
            raise ValueError("Input DataFrame is empty.")

        target_series = dataframe[target_column].copy()
        if target_series.isnull().any():
            warnings.warn(
                f"Target column '{target_column}' contains NaN values. Filling with empty strings."
            )
            target_series = target_series.fillna('').astype(str)

        pattern_feature = self.extract_pattern_feature(target_series.tolist())
        rule_feature = self.extract_rule_feature(dataframe, target_column)

        feature_parts = [pattern_feature, rule_feature]
        feature_dataframe = pandas.concat(feature_parts, axis=1)

        scaled_features = self.standard_scaler.fit_transform(feature_dataframe)
        feature_dataframe = pandas.DataFrame(scaled_features, columns=feature_dataframe.columns)

        return feature_dataframe
