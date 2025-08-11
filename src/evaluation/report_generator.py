"""
报告生成器
生成详细的评估报告
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ReportGenerator:
    """评估报告生成器"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # 默认使用项目根目录下的reports文件夹
            current_dir = Path(__file__).parent.parent.parent
            self.output_dir = current_dir / "reports"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True)

    def generate_detailed_report(self, evaluation_result: Dict[str, Any],
                               format_type: str = 'json') -> str:
        """
        生成详细评估报告

        Args:
            evaluation_result: 评估结果
            format_type: 报告格式 ('json', 'html', 'markdown')

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = evaluation_result.get('metadata', {})
        dataset_name = metadata.get('dataset_name', 'unknown')
        model_name = metadata.get('model_name', 'unknown')

        filename = f"evaluation_report_{dataset_name}_{model_name}_{timestamp}"

        if format_type == 'json':
            return self._generate_json_report(evaluation_result, filename)
        elif format_type == 'html':
            return self._generate_html_report(evaluation_result, filename)
        elif format_type == 'markdown':
            return self._generate_markdown_report(evaluation_result, filename)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_json_report(self, evaluation_result: Dict[str, Any], filename: str) -> str:
        """生成JSON格式报告"""
        filepath = self.output_dir / f"{filename}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False, default=str)

        return str(filepath)

    def _generate_html_report(self, evaluation_result: Dict[str, Any], filename: str) -> str:
        """生成HTML格式报告"""
        filepath = self.output_dir / f"{filename}.html"

        html_content = self._build_html_content(evaluation_result)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)

    def _generate_markdown_report(self, evaluation_result: Dict[str, Any], filename: str) -> str:
        """生成Markdown格式报告"""
        filepath = self.output_dir / f"{filename}.md"

        md_content = self._build_markdown_content(evaluation_result)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return str(filepath)

    def _build_html_content(self, evaluation_result: Dict[str, Any]) -> str:
        """构建HTML报告内容"""
        metadata = evaluation_result.get('metadata', {})
        metrics = evaluation_result.get('metrics', {})
        analysis = evaluation_result.get('analysis', {})

        html = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>错误检测评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 8px; }}
        .metrics {{ background-color: #e9f7ef; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        .analysis {{ background-color: #fef9e7; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .performance-excellent {{ color: #28a745; font-weight: bold; }}
        .performance-good {{ color: #007bff; font-weight: bold; }}
        .performance-fair {{ color: #ffc107; font-weight: bold; }}
        .performance-poor {{ color: #fd7e14; font-weight: bold; }}
        .performance-very-poor {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>错误检测评估报告</h1>
        <p><strong>数据集:</strong> {metadata.get('dataset_name', 'N/A')}</p>
        <p><strong>模型:</strong> {metadata.get('model_name', 'N/A')}</p>
        <p><strong>评估时间:</strong> {metadata.get('evaluation_timestamp', 'N/A')}</p>
        <p><strong>数据规模:</strong> {metadata.get('data_shape', 'N/A')}</p>
    </div>
"""

        # 整体指标
        overall_metrics = metrics.get('overall', {})
        if overall_metrics:
            performance_level = analysis.get('performance_level', 'unknown')
            performance_class = f"performance-{performance_level.replace('_', '-')}"

            html += f"""
    <div class="metrics">
        <h2>整体性能指标</h2>
        <p><strong>性能等级:</strong> <span class="{performance_class}">{performance_level.upper()}</span></p>
        <table>
            <tr><th>指标</th><th>值</th></tr>
            <tr><td>准确率 (Accuracy)</td><td>{overall_metrics.get('accuracy', 0):.4f}</td></tr>
            <tr><td>精确率 (Precision)</td><td>{overall_metrics.get('precision', 0):.4f}</td></tr>
            <tr><td>召回率 (Recall)</td><td>{overall_metrics.get('recall', 0):.4f}</td></tr>
            <tr><td>F1分数 (F1-Score)</td><td>{overall_metrics.get('f1_score', 0):.4f}</td></tr>
            <tr><td>真正例 (TP)</td><td>{overall_metrics.get('true_positives', 0)}</td></tr>
            <tr><td>假正例 (FP)</td><td>{overall_metrics.get('false_positives', 0)}</td></tr>
            <tr><td>假负例 (FN)</td><td>{overall_metrics.get('false_negatives', 0)}</td></tr>
            <tr><td>真负例 (TN)</td><td>{overall_metrics.get('true_negatives', 0)}</td></tr>
        </table>
    </div>
"""

        # 列级别指标
        column_metrics = metrics.get('column_wise', {})
        if column_metrics:
            html += """
    <div class="metrics">
        <h2>列级别性能指标</h2>
        <table>
            <tr><th>列名</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1分数</th></tr>
"""
            for col_name, col_metrics in column_metrics.items():
                html += f"""
            <tr>
                <td>{col_name}</td>
                <td>{col_metrics.get('accuracy', 0):.4f}</td>
                <td>{col_metrics.get('precision', 0):.4f}</td>
                <td>{col_metrics.get('recall', 0):.4f}</td>
                <td>{col_metrics.get('f1_score', 0):.4f}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""

        # 分析和建议
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            html += """
    <div class="analysis">
        <h2>分析和建议</h2>
        <ul>
"""
            for rec in recommendations:
                html += f"            <li>{rec}</li>\n"

            html += """
        </ul>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def _build_markdown_content(self, evaluation_result: Dict[str, Any]) -> str:
        """构建Markdown报告内容"""
        metadata = evaluation_result.get('metadata', {})
        metrics = evaluation_result.get('metrics', {})
        analysis = evaluation_result.get('analysis', {})

        md = f"""# 错误检测评估报告

## 基本信息
- **数据集**: {metadata.get('dataset_name', 'N/A')}
- **模型**: {metadata.get('model_name', 'N/A')}
- **评估时间**: {metadata.get('evaluation_timestamp', 'N/A')}
- **数据规模**: {metadata.get('data_shape', 'N/A')}

"""

        # 整体指标
        overall_metrics = metrics.get('overall', {})
        if overall_metrics:
            performance_level = analysis.get('performance_level', 'unknown')

            md += f"""## 整体性能指标

**性能等级**: {performance_level.upper()}

| 指标 | 值 |
|------|------|
| 准确率 (Accuracy) | {overall_metrics.get('accuracy', 0):.4f} |
| 精确率 (Precision) | {overall_metrics.get('precision', 0):.4f} |
| 召回率 (Recall) | {overall_metrics.get('recall', 0):.4f} |
| F1分数 (F1-Score) | {overall_metrics.get('f1_score', 0):.4f} |
| 真正例 (TP) | {overall_metrics.get('true_positives', 0)} |
| 假正例 (FP) | {overall_metrics.get('false_positives', 0)} |
| 假负例 (FN) | {overall_metrics.get('false_negatives', 0)} |
| 真负例 (TN) | {overall_metrics.get('true_negatives', 0)} |

"""

        # 列级别指标
        column_metrics = metrics.get('column_wise', {})
        if column_metrics:
            md += """## 列级别性能指标

| 列名 | 准确率 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|--------|
"""
            for col_name, col_metrics in column_metrics.items():
                md += f"| {col_name} | {col_metrics.get('accuracy', 0):.4f} | {col_metrics.get('precision', 0):.4f} | {col_metrics.get('recall', 0):.4f} | {col_metrics.get('f1_score', 0):.4f} |\n"

            md += "\n"

        # 分析和建议
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            md += "## 分析和建议\n\n"
            for i, rec in enumerate(recommendations, 1):
                md += f"{i}. {rec}\n"
            md += "\n"

        return md

    def generate_comparison_report(self, comparison_result: Dict[str, Any],
                                 format_type: str = 'html') -> str:
        """生成模型比较报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_report_{timestamp}"

        if format_type == 'html':
            return self._generate_comparison_html(comparison_result, filename)
        elif format_type == 'markdown':
            return self._generate_comparison_markdown(comparison_result, filename)
        else:
            return self._generate_json_report(comparison_result, filename)

    def _generate_comparison_html(self, comparison_result: Dict[str, Any], filename: str) -> str:
        """生成HTML格式的比较报告"""
        filepath = self.output_dir / f"{filename}.html"

        metrics_comparison = comparison_result.get('metrics_comparison', {})
        performance_ranking = comparison_result.get('performance_ranking', [])

        html = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>模型比较报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .rank-1 { background-color: #d4edda; }
        .rank-2 { background-color: #d1ecf1; }
        .rank-3 { background-color: #ffeaa7; }
    </style>
</head>
<body>
    <h1>模型比较报告</h1>
    
    <h2>性能排名</h2>
    <table>
        <tr><th>排名</th><th>模型</th><th>F1分数</th></tr>
"""

        for i, model_info in enumerate(performance_ranking, 1):
            rank_class = f"rank-{min(i, 3)}"
            html += f"""
        <tr class="{rank_class}">
            <td>{i}</td>
            <td>{model_info['model_name']}</td>
            <td>{model_info['f1_score']:.4f}</td>
        </tr>
"""

        html += """
    </table>
    
    <h2>详细指标比较</h2>
    <table>
        <tr><th>模型</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1分数</th></tr>
"""

        for model_name, model_metrics in metrics_comparison.items():
            html += f"""
        <tr>
            <td>{model_name}</td>
            <td>{model_metrics.get('accuracy', 0):.4f}</td>
            <td>{model_metrics.get('precision', 0):.4f}</td>
            <td>{model_metrics.get('recall', 0):.4f}</td>
            <td>{model_metrics.get('f1_score', 0):.4f}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(filepath)
