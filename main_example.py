"""
主程序示例
展示如何使用src目录下重构后的各个模块
"""
import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data import DatasetLoader, DataManager
from llm import LLMFactory, OllamaLLM, OpenAILLM
from detection import ErrorDetector, PromptManager
from evaluation import Evaluator, ReportGenerator


def main():
    """主函数：完整的错误检测和评估流程"""

    # 1. 数据准备
    print("=== 数据加载 ===")
    loader = DatasetLoader()
    data_manager = DataManager(loader)

    # 查看可用数据集
    datasets = loader.get_available_datasets()
    print(f"可用数据集: {datasets}")

    # 选择数据集
    dataset_name = "hospital"  # 可以改为其他数据集
    print(f"使用数据集: {dataset_name}")

    # 准备检测数据
    prepared_data = data_manager.prepare_data_for_detection(dataset_name, sample_size=50)
    clean_data = prepared_data['clean_data']
    dirty_data = prepared_data['dirty_data']
    error_labels = prepared_data['error_labels']
    rules = prepared_data['rules']

    print(f"数据形状: {dirty_data.shape}")
    print(f"错误数量: {error_labels.sum().sum()}")

    # 2. LLM配置
    print("\n=== LLM配置 ===")

    # 方式1: 使用工厂模式创建LLM
    llm = LLMFactory.create_ollama_llm("llama3.1:8b")

    # 方式2: 直接创建（如果使用OpenAI）
    # llm = OpenAILLM(
    #     model_name="gpt-3.5-turbo",
    #     api_key="your-api-key",
    #     base_url="https://api.openai.com/v1"
    # )

    print(f"使用模型: {llm.get_model_name()}")

    # 3. 错误检测
    print("\n=== 错误检测 ===")

    # 创建检测器
    prompt_manager = PromptManager()
    detector = ErrorDetector(llm, prompt_manager)

    # 执行检测
    detection_modes = ["zero_shot", "rule_based"]

    for mode in detection_modes:
        print(f"\n执行 {mode} 检测...")

        predictions = detector.detect_errors(
            data=dirty_data,
            rules=rules,
            detection_mode=mode,
            batch_size=5
        )

        # 4. 结果评估
        print(f"\n=== {mode} 评估结果 ===")

        evaluator = Evaluator()
        evaluation_result = evaluator.evaluate_detection_results(
            y_true=error_labels,
            y_pred=predictions,
            dataset_name=dataset_name,
            model_name=llm.get_model_name(),
            detection_config={'mode': mode, 'batch_size': 5}
        )

        # 显示摘要
        summary = evaluator.get_evaluation_summary()
        print(f"准确率: {summary['accuracy']:.4f}")
        print(f"精确率: {summary['precision']:.4f}")
        print(f"召回率: {summary['recall']:.4f}")
        print(f"F1分数: {summary['f1_score']:.4f}")
        print(f"性能等级: {summary['performance_level']}")

        # 生成报告
        report_generator = ReportGenerator()
        report_path = report_generator.generate_detailed_report(
            evaluation_result,
            format_type='html'
        )
        print(f"详细报告已生成: {report_path}")

        # 显示建议
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print("\n改进建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

    # 5. 模型比较（如果有多个评估结果）
    print("\n=== 模型比较 ===")

    if len(evaluator.evaluation_history) >= 2:
        comparison = evaluator.compare_evaluations([0, 1])
        print("性能排名:")
        for i, model_info in enumerate(comparison['performance_ranking'], 1):
            print(f"  {i}. {model_info['model_name']}: F1={model_info['f1_score']:.4f}")

        # 生成比较报告
        comparison_report_path = report_generator.generate_comparison_report(
            comparison, format_type='html'
        )
        print(f"比较报告已生成: {comparison_report_path}")


def demo_data_management():
    """数据管理模块演示"""
    print("=== 数据管理演示 ===")

    loader = DatasetLoader()
    data_manager = DataManager(loader)

    # 获取数据集信息
    datasets = loader.get_available_datasets()
    for dataset_name in datasets[:2]:  # 演示前2个数据集
        print(f"\n数据集: {dataset_name}")
        info = loader.get_dataset_info(dataset_name)
        print(f"  - 有clean数据: {info['has_clean_data']}")
        print(f"  - 有dirty数据: {info['has_dirty_data']}")
        print(f"  - 有规则文件: {info['has_rules']}")
        if 'clean_shape' in info:
            print(f"  - 数据形状: {info['clean_shape']}")
            print(f"  - 列数: {len(info['columns'])}")


def demo_llm_usage():
    """LLM模块使用演示"""
    print("=== LLM模块演示 ===")

    # 显示支持的提供商和模型
    factory = LLMFactory()
    print(f"支持的提供商: {factory.list_supported_providers()}")
    print(f"常用模型: {factory.list_common_models()[:5]}")  # 显示前5个

    # 创建LLM实例
    try:
        llm = factory.create_ollama_llm("llama3.1:8b")
        print(f"成功创建LLM: {llm.get_model_name()}")

        # 简单测试
        response = llm.generate("Hello, how are you?")
        print(f"测试响应: {response[:100]}...")

    except Exception as e:
        print(f"LLM创建失败: {e}")


if __name__ == "__main__":
    # 运行完整流程
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        print("\n运行演示模块...")

        # 运行各模块演示
        demo_data_management()
        demo_llm_usage()

        print("\n请确保:")
        print("1. 已安装required包: pip install -r requirements.txt")
        print("2. Ollama服务已启动，并下载了相应模型")
        print("3. datasets目录包含数据文件")
