"""
特征提取器
为错误检测提取数据特征
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from collections import Counter


class FeatureExtractor:
    """数据特征提取器"""

    def __init__(self):
        self.feature_cache = {}

    def extract_column_features(self, series: pd.Series) -> Dict[str, Any]:
        """提取单列的特征"""
        features = {}

        # 基本统计特征
        features.update(self._extract_basic_stats(series))

        # 数据类型特征
        features.update(self._extract_data_type_features(series))

        # 模式特征
        features.update(self._extract_pattern_features(series))

        # 异常值特征
        features.update(self._extract_outlier_features(series))

        # 完整性特征
        features.update(self._extract_completeness_features(series))

        return features

    def _extract_basic_stats(self, series: pd.Series) -> Dict[str, Any]:
        """提取基本统计特征"""
        stats = {
            'count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
            'most_common_value': None,
            'most_common_count': 0
        }

        # 最常见值
        if stats['non_null_count'] > 0:
            value_counts = series.value_counts()
            stats['most_common_value'] = value_counts.index[0]
            stats['most_common_count'] = value_counts.iloc[0]

        # 如果是数值型，添加数值统计
        numeric_series = pd.to_numeric(series, errors='coerce')
        if not numeric_series.isnull().all():
            stats.update({
                'mean': numeric_series.mean(),
                'std': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'median': numeric_series.median()
            })

        return stats

    def _extract_data_type_features(self, series: pd.Series) -> Dict[str, Any]:
        """提取数据类型特征"""
        features = {
            'pandas_dtype': str(series.dtype),
            'is_numeric': pd.api.types.is_numeric_dtype(series),
            'is_string': pd.api.types.is_string_dtype(series),
            'is_datetime': pd.api.types.is_datetime64_any_dtype(series)
        }

        # 分析实际数据类型分布
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            type_counts = {}
            for value in non_null_values.head(100):  # 采样前100个值
                value_type = self._infer_value_type(value)
                type_counts[value_type] = type_counts.get(value_type, 0) + 1

            features['inferred_types'] = type_counts
            features['dominant_type'] = max(type_counts, key=type_counts.get) if type_counts else 'unknown'

        return features

    def _extract_pattern_features(self, series: pd.Series) -> Dict[str, Any]:
        """提取模式特征"""
        features = {
            'pattern_diversity': 0,
            'common_patterns': [],
            'has_consistent_format': False
        }

        non_null_values = series.dropna().astype(str)
        if len(non_null_values) == 0:
            return features

        # 提取字符模式
        patterns = []
        for value in non_null_values.head(100):  # 采样
            pattern = self._extract_char_pattern(str(value))
            patterns.append(pattern)

        pattern_counts = Counter(patterns)
        features['pattern_diversity'] = len(pattern_counts)
        features['common_patterns'] = list(pattern_counts.most_common(5))

        # 判断格式一致性
        if len(pattern_counts) == 1:
            features['has_consistent_format'] = True
        elif len(pattern_counts) <= 3 and len(non_null_values) > 10:
            # 如果模式种类很少且数据量不小，认为格式相对一致
            most_common_ratio = pattern_counts.most_common(1)[0][1] / len(patterns)
            features['has_consistent_format'] = most_common_ratio > 0.8

        return features

    def _extract_outlier_features(self, series: pd.Series) -> Dict[str, Any]:
        """提取异常值特征"""
        features = {
            'potential_outliers_count': 0,
            'length_outliers': 0,
            'format_outliers': 0
        }

        non_null_values = series.dropna()
        if len(non_null_values) < 2:
            return features

        # 长度异常值检测
        if pd.api.types.is_string_dtype(series):
            lengths = non_null_values.astype(str).str.len()
            q1, q3 = lengths.quantile([0.25, 0.75])
            iqr = q3 - q1
            length_outliers = lengths[(lengths < q1 - 1.5 * iqr) | (lengths > q3 + 1.5 * iqr)]
            features['length_outliers'] = len(length_outliers)

        # 数值异常值检测
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) > 2:
            q1, q3 = numeric_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                numeric_outliers = numeric_series[(numeric_series < q1 - 1.5 * iqr) |
                                                (numeric_series > q3 + 1.5 * iqr)]
                features['potential_outliers_count'] = len(numeric_outliers)

        return features

    def _extract_completeness_features(self, series: pd.Series) -> Dict[str, Any]:
        """提取完整性特征"""
        features = {
            'completeness_ratio': series.count() / len(series),
            'has_empty_strings': False,
            'has_whitespace_only': False
        }

        if pd.api.types.is_string_dtype(series):
            string_values = series.dropna().astype(str)
            features['has_empty_strings'] = (string_values == '').any()
            features['has_whitespace_only'] = string_values.str.strip().eq('').any()

        return features

    def _infer_value_type(self, value) -> str:
        """推断单个值的类型"""
        if pd.isna(value):
            return 'null'

        value_str = str(value).strip()

        # 检查是否为数字
        if value_str.replace('.', '').replace('-', '').isdigit():
            if '.' in value_str:
                return 'float'
            else:
                return 'integer'

        # 检查是否为日期
        if self._looks_like_date(value_str):
            return 'date'

        # 检查是否为邮箱
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_str):
            return 'email'

        # 检查是否为电话号码
        if re.match(r'^[\d\s\-\+\(\)]+$', value_str) and len(value_str) > 7:
            return 'phone'

        return 'string'

    def _extract_char_pattern(self, value: str) -> str:
        """提取字符模式"""
        if not value:
            return 'empty'

        pattern = ''
        for char in value:
            if char.isdigit():
                pattern += 'D'
            elif char.isalpha():
                if char.isupper():
                    pattern += 'U'
                else:
                    pattern += 'L'
            elif char.isspace():
                pattern += 'S'
            else:
                pattern += 'P'  # Punctuation/Special

        # 压缩连续相同的字符
        compressed = ''
        prev_char = ''
        count = 0

        for char in pattern:
            if char == prev_char:
                count += 1
            else:
                if prev_char and count > 2:
                    compressed += f"{prev_char}{count}"
                elif prev_char:
                    compressed += prev_char * count
                prev_char = char
                count = 1

        # 处理最后一组
        if prev_char and count > 2:
            compressed += f"{prev_char}{count}"
        elif prev_char:
            compressed += prev_char * count

        return compressed or pattern

    def _looks_like_date(self, value: str) -> bool:
        """判断是否像日期"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]

        return any(re.match(pattern, value) for pattern in date_patterns)
