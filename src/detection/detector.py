"""
错误检测器
基于LLM的数据错误检测主要逻辑
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..llm.base_llm import BaseLLM
from .prompt_manager import PromptManager
from .feature_extractor import FeatureExtractor
from .detection_utils import DetectionUtils


class ErrorDetector:
    """基于LLM的错误检测器"""

    def __init__(self, llm: BaseLLM, prompt_manager: PromptManager = None):
        self.llm = llm
        self.prompt_manager = prompt_manager if prompt_manager else PromptManager()
        self.feature_extractor = FeatureExtractor()
        self.utils = DetectionUtils()

    def detect_errors(self, data: pd.DataFrame, rules: Dict[str, Any] = None,
                     detection_mode: str = 'zero_shot',
                     few_shot_examples: List[Dict] = None,
                     batch_size: int = 10) -> pd.DataFrame:
        """
        检测数据中的错误

        Args:
            data: 待检测的数据
            rules: 数据质量规则
            detection_mode: 检测模式 ('zero_shot', 'few_shot', 'rule_based')
            few_shot_examples: few-shot示例
            batch_size: 批处理大小

        Returns:
            错误标签DataFrame，1表示错误，0表示正确
        """
        error_labels = pd.DataFrame(0, index=data.index, columns=data.columns)

        # 按列进行错误检测
        for column in data.columns:
            column_errors = self._detect_column_errors(
                data[column], column, rules, detection_mode,
                few_shot_examples, batch_size
            )
            error_labels[column] = column_errors

        return error_labels

    def _detect_column_errors(self, series: pd.Series, column_name: str,
                             rules: Dict[str, Any] = None,
                             detection_mode: str = 'zero_shot',
                             few_shot_examples: List[Dict] = None,
                             batch_size: int = 10) -> pd.Series:
        """检测单列的错误"""

        # 提取列的规则信息
        column_rule = self._get_column_rule(column_name, rules)

        # 分批处理数据
        batches = self.utils.create_batches(series, batch_size)
        all_predictions = []

        for batch_data, batch_indices in batches:
            # 生成提示词
            if detection_mode == 'zero_shot':
                prompt = self.prompt_manager.generate_zero_shot_prompt(
                    batch_data, column_name, column_rule
                )
            elif detection_mode == 'few_shot':
                prompt = self.prompt_manager.generate_few_shot_prompt(
                    batch_data, column_name, column_rule, few_shot_examples
                )
            elif detection_mode == 'rule_based':
                prompt = self.prompt_manager.generate_rule_based_prompt(
                    batch_data, column_name, column_rule
                )
            else:
                raise ValueError(f"Unsupported detection mode: {detection_mode}")

            # 调用LLM
            try:
                response = self.llm.generate(prompt)
                batch_predictions = self._parse_llm_response(response, len(batch_data))
            except Exception as e:
                print(f"LLM调用失败: {str(e)}")
                # 默认预测为无错误
                batch_predictions = [0] * len(batch_data)

            all_predictions.extend(batch_predictions)

        return pd.Series(all_predictions, index=series.index)

    def _get_column_rule(self, column_name: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """获取指定列的规则信息"""
        if not rules or 'columns' not in rules:
            return {}

        for col_rule in rules['columns']:
            if col_rule.get('name') == column_name:
                return col_rule

        return {}

    def _parse_llm_response(self, response: str, expected_length: int) -> List[int]:
        """解析LLM响应，提取错误标签"""
        try:
            # 尝试解析JSON格式的响应
            labels = self.utils.extract_labels_from_json(response)
            if labels and len(labels) == expected_length:
                return labels
        except:
            pass

        try:
            # 尝试解析简单的数字列表
            labels = self.utils.extract_labels_from_text(response)
            if labels and len(labels) == expected_length:
                return labels
        except:
            pass

        # 如果解析失败，返回默认值
        print(f"警告: 无法解析LLM响应，使用默认值。响应内容: {response[:100]}...")
        return [0] * expected_length

    def detect_errors_with_confidence(self, data: pd.DataFrame,
                                    rules: Dict[str, Any] = None,
                                    detection_mode: str = 'zero_shot',
                                    threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        检测错误并返回置信度

        Returns:
            (error_labels, confidence_scores)
        """
        # 这里可以实现更复杂的置信度计算逻辑
        error_labels = self.detect_errors(data, rules, detection_mode)

        # 简单的置信度计算（可以根据需要改进）
        confidence_scores = pd.DataFrame(
            np.where(error_labels == 1, 0.8, 0.9),  # 错误预测置信度0.8，正确预测0.9
            index=data.index,
            columns=data.columns
        )

        return error_labels, confidence_scores

    def evaluate_detection_quality(self, predictions: pd.DataFrame,
                                 ground_truth: pd.DataFrame) -> Dict[str, Any]:
        """评估检测质量"""
        from ..evaluation.metrics import calculate_metrics
        return calculate_metrics(ground_truth, predictions)
