"""
检测工具函数
提供错误检测过程中的辅助功能
"""
import json
import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


class DetectionUtils:
    """错误检测工具类"""

    def __init__(self):
        pass

    def create_batches(self, series: pd.Series, batch_size: int) -> List[Tuple[List, List]]:
        """
        将数据分批处理

        Args:
            series: 输入数据序列
            batch_size: 批次大小

        Returns:
            [(batch_data, batch_indices), ...]
        """
        batches = []
        data_list = series.tolist()
        indices_list = series.index.tolist()

        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i + batch_size]
            batch_indices = indices_list[i:i + batch_size]
            batches.append((batch_data, batch_indices))

        return batches

    def extract_labels_from_json(self, response: str) -> Optional[List[int]]:
        """
        从JSON格式的响应中提取标签

        Args:
            response: LLM响应内容

        Returns:
            标签列表或None
        """
        try:
            # 查找JSON模式
            json_pattern = r'(\{\s*"labels"\s*:\s*\[[\s\S]*?\]\s*\})'
            match = re.search(json_pattern, response)

            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                if 'labels' in data and isinstance(data['labels'], list):
                    # 确保所有标签都是0或1
                    labels = [int(label) if label in [0, 1] else 0 for label in data['labels']]
                    return labels

            # 尝试直接解析整个响应为JSON
            data = json.loads(response.strip())
            if 'labels' in data and isinstance(data['labels'], list):
                labels = [int(label) if label in [0, 1] else 0 for label in data['labels']]
                return labels

        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        return None

    def extract_labels_from_text(self, response: str) -> Optional[List[int]]:
        """
        从文本响应中提取标签

        Args:
            response: LLM响应内容

        Returns:
            标签列表或None
        """
        try:
            # 查找数字序列
            numbers = re.findall(r'\b[01]\b', response)
            if numbers:
                return [int(num) for num in numbers]

            # 查找带逗号分隔的数字
            comma_pattern = r'\[?\s*([01](?:\s*,\s*[01])*)\s*\]?'
            match = re.search(comma_pattern, response)
            if match:
                numbers_str = match.group(1)
                numbers = [int(num.strip()) for num in numbers_str.split(',')]
                return numbers

        except (ValueError, AttributeError):
            pass

        return None

    def validate_labels(self, labels: List[int], expected_length: int) -> bool:
        """
        验证标签列表的有效性

        Args:
            labels: 标签列表
            expected_length: 期望长度

        Returns:
            是否有效
        """
        if not isinstance(labels, list):
            return False

        if len(labels) != expected_length:
            return False

        # 检查所有标签都是0或1
        return all(label in [0, 1] for label in labels)

    def chunk_data_by_size(self, data: List[Any], max_chars: int = 2000) -> List[List[Any]]:
        """
        按字符长度分割数据

        Args:
            data: 数据列表
            max_chars: 最大字符数

        Returns:
            分割后的数据块列表
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for item in data:
            item_size = len(str(item))

            if current_size + item_size > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [item]
                current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def format_data_for_prompt(self, data: List[Any]) -> str:
        """
        格式化数据用于提示词

        Args:
            data: 数据列表

        Returns:
            格式化的字符串
        """
        formatted_lines = []
        for i, value in enumerate(data, 1):
            # 处理特殊值
            if pd.isna(value):
                display_value = "[NULL]"
            elif value == "":
                display_value = "[EMPTY]"
            elif isinstance(value, str) and value.isspace():
                display_value = "[WHITESPACE]"
            else:
                display_value = str(value)

            formatted_lines.append(f"{i}. {display_value}")

        return "\n".join(formatted_lines)

    def select_few_shot_examples(self, clean_data: pd.Series, dirty_data: pd.Series,
                               error_labels: pd.Series, n_examples: int = 5) -> List[Dict]:
        """
        选择few-shot示例

        Args:
            clean_data: 干净数据
            dirty_data: 脏数据
            error_labels: 错误标签
            n_examples: 示例数量

        Returns:
            示例列表
        """
        examples = []

        # 选择正负例的平衡
        positive_indices = error_labels[error_labels == 1].index
        negative_indices = error_labels[error_labels == 0].index

        # 选择一半正例，一半负例
        n_positive = min(n_examples // 2, len(positive_indices))
        n_negative = min(n_examples - n_positive, len(negative_indices))

        # 随机选择正例
        if n_positive > 0:
            selected_positive = positive_indices[:n_positive]
            for idx in selected_positive:
                examples.append({
                    'value': dirty_data.loc[idx],
                    'clean_value': clean_data.loc[idx],
                    'label': 1,
                    'explanation': f"Error: '{dirty_data.loc[idx]}' should be '{clean_data.loc[idx]}'"
                })

        # 随机选择负例
        if n_negative > 0:
            selected_negative = negative_indices[:n_negative]
            for idx in selected_negative:
                examples.append({
                    'value': dirty_data.loc[idx],
                    'clean_value': clean_data.loc[idx],
                    'label': 0,
                    'explanation': "Correct value"
                })

        return examples

    def log_detection_info(self, message: str, level: str = "INFO"):
        """
        记录检测信息

        Args:
            message: 日志消息
            level: 日志级别
        """
        print(f"[{level}] {message}")

    def calculate_detection_statistics(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        计算检测统计信息

        Args:
            predictions: 预测结果DataFrame

        Returns:
            统计信息字典
        """
        stats = {
            'total_cells': predictions.size,
            'total_errors_detected': predictions.sum().sum(),
            'error_rate': predictions.sum().sum() / predictions.size,
            'columns_with_errors': (predictions.sum() > 0).sum(),
            'column_error_rates': (predictions.sum() / len(predictions)).to_dict()
        }

        return stats
