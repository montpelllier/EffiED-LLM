"""
测试运行器
统一运行所有模块的测试
"""
import unittest
import sys
from pathlib import Path

# 添加路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))
sys.path.insert(0, str(test_dir.parent))

# 导入所有测试模块
from test_data import TestDatasetLoader, TestDataManager
from test_detection_features import TestFeatureExtractor
from test_detection_prompt import TestPromptManager
from test_evaluation import TestMetricsCalculator, TestEvaluator, TestReportGenerator
from test_llm import TestBaseLLM, TestOllamaLLM, TestOpenAILLM, TestLLMFactory


def create_test_suite():
    """创建完整的测试套件"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加data模块测试
    print("添加data模块测试...")
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDataManager))

    # 添加detection模块测试
    print("添加detection模块测试...")
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptManager))

    # 添加evaluation模块测试
    print("添加evaluation模块测试...")
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestReportGenerator))

    # 添加llm模块测试
    print("添加llm模块测试...")
    suite.addTests(loader.loadTestsFromTestCase(TestBaseLLM))
    suite.addTests(loader.loadTestsFromTestCase(TestOllamaLLM))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenAILLM))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFactory))

    return suite


def run_specific_module_tests(module_name):
    """运行特定模块的测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if module_name == 'data':
        print(f"=== 运行{module_name}模块测试 ===")
        suite.addTests(loader.loadTestsFromTestCase(TestDatasetLoader))
        suite.addTests(loader.loadTestsFromTestCase(TestDataManager))

    elif module_name == 'detection':
        print(f"=== 运行{module_name}模块测试 ===")
        suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtractor))
        suite.addTests(loader.loadTestsFromTestCase(TestPromptManager))

    elif module_name == 'evaluation':
        print(f"=== 运行{module_name}模块测试 ===")
        suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculator))
        suite.addTests(loader.loadTestsFromTestCase(TestEvaluator))
        suite.addTests(loader.loadTestsFromTestCase(TestReportGenerator))

    elif module_name == 'llm':
        print(f"=== 运行{module_name}模块测试 ===")
        suite.addTests(loader.loadTestsFromTestCase(TestBaseLLM))
        suite.addTests(loader.loadTestsFromTestCase(TestOllamaLLM))
        suite.addTests(loader.loadTestsFromTestCase(TestOpenAILLM))
        suite.addTests(loader.loadTestsFromTestCase(TestLLMFactory))

    else:
        print(f"未知模块: {module_name}")
        return

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{module_name}模块测试完成:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    return result


def main():
    """主测试入口"""
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        run_specific_module_tests(module_name)
    else:
        print("=== 运行所有模块测试 ===")

        # 创建测试套件
        suite = create_test_suite()

        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # 输出总结
        print("\n" + "="*50)
        print("测试总结:")
        print(f"总计运行测试: {result.testsRun}")
        print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"失败: {len(result.failures)}")
        print(f"错误: {len(result.errors)}")

        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")

        # 返回退出码
        return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    print("测试模块启动...")
    print("使用方法:")
    print("  python run_tests.py           # 运行所有测试")
    print("  python run_tests.py data      # 只运行data模块测试")
    print("  python run_tests.py detection # 只运行detection模块测试")
    print("  python run_tests.py evaluation# 只运行evaluation模块测试")
    print("  python run_tests.py llm       # 只运行llm模块测试")
    print()

    exit_code = main()
    sys.exit(exit_code)
