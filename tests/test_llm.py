"""
测试llm模块的功能
包括BaseLLM、OllamaLLM、OpenAILLM和LLMFactory的测试
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# 添加路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir.parent / "src"))

from llm import BaseLLM, OllamaLLM, OpenAILLM, LLMFactory


class TestBaseLLM(unittest.TestCase):
    """测试BaseLLM基类"""

    def test_base_llm_instantiation(self):
        """测试基类不能直接实例化"""
        with self.assertRaises(TypeError):
            BaseLLM("test_model")

    def test_base_llm_abstract_methods(self):
        """测试抽象方法必须被实现"""

        class IncompleteLLM(BaseLLM):
            # 只实现一个抽象方法
            def chat(self, messages, **kwargs):
                return "test"

        with self.assertRaises(TypeError):
            IncompleteLLM("test_model")

    def test_base_llm_concrete_methods(self):
        """测试具体方法的实现"""

        class CompleteLLM(BaseLLM):
            def chat(self, messages, **kwargs):
                return "test chat"

            def generate(self, prompt, **kwargs):
                return "test generate"

        llm = CompleteLLM("test_model", config_param="test_value")

        # 测试基类方法
        self.assertEqual(llm.get_model_name(), "test_model")

        config = llm.get_config()
        self.assertEqual(config['config_param'], "test_value")

        # 测试设置配置
        llm.set_config(new_param="new_value")
        updated_config = llm.get_config()
        self.assertEqual(updated_config['new_param'], "new_value")


class TestOllamaLLM(unittest.TestCase):
    """测试OllamaLLM类"""

    def setUp(self):
        """设置测试环境"""
        self.model_name = "llama3.1:8b"

    @patch('llm.ollama_llm.ollama')
    def test_ollama_llm_initialization(self, mock_ollama):
        """测试Ollama LLM初始化"""
        llm = OllamaLLM(self.model_name)

        self.assertEqual(llm.get_model_name(), self.model_name)
        self.assertIsNone(llm.host)

        # 测试带host的初始化
        llm_with_host = OllamaLLM(self.model_name, host="http://localhost:11434")
        self.assertEqual(llm_with_host.host, "http://localhost:11434")

    @patch('llm.ollama_llm.ollama')
    def test_chat_method(self, mock_ollama):
        """测试chat方法"""
        # 模拟ollama.chat返回值
        mock_response = {
            'message': {
                'content': 'This is a test response'
            }
        }
        mock_ollama.chat.return_value = mock_response

        llm = OllamaLLM(self.model_name)
        messages = [{"role": "user", "content": "Hello"}]

        response = llm.chat(messages)

        self.assertEqual(response, 'This is a test response')
        mock_ollama.chat.assert_called_once_with(
            model=self.model_name,
            messages=messages
        )

    @patch('llm.ollama_llm.ollama')
    def test_generate_method(self, mock_ollama):
        """测试generate方法"""
        # 模拟ollama.generate返回值
        mock_response = {
            'response': 'Generated text response'
        }
        mock_ollama.generate.return_value = mock_response

        llm = OllamaLLM(self.model_name)
        prompt = "Generate some text"

        response = llm.generate(prompt)

        self.assertEqual(response, 'Generated text response')
        mock_ollama.generate.assert_called_once_with(
            model=self.model_name,
            prompt=prompt
        )

    @patch('llm.ollama_llm.ollama')
    def test_is_model_available(self, mock_ollama):
        """测试模型可用性检查"""
        # 模拟模型列表
        mock_models = {
            'models': [
                {'name': 'llama3.1:8b'},
                {'name': 'qwen2.5:7b'}
            ]
        }
        mock_ollama.list.return_value = mock_models

        llm = OllamaLLM("llama3.1:8b")
        self.assertTrue(llm.is_model_available())

        llm_unavailable = OllamaLLM("nonexistent:model")
        self.assertFalse(llm_unavailable.is_model_available())

    @patch('llm.ollama_llm.ollama')
    def test_pull_model(self, mock_ollama):
        """测试模型拉取"""
        llm = OllamaLLM(self.model_name)

        # 成功拉取
        mock_ollama.pull.return_value = None
        result = llm.pull_model()
        self.assertTrue(result)
        mock_ollama.pull.assert_called_with(self.model_name)

        # 拉取失败
        mock_ollama.pull.side_effect = Exception("Pull failed")
        result = llm.pull_model()
        self.assertFalse(result)

    @patch('llm.ollama_llm.ollama')
    def test_error_handling(self, mock_ollama):
        """测试错误处理"""
        llm = OllamaLLM(self.model_name)

        # 测试chat方法错误处理
        mock_ollama.chat.side_effect = Exception("Connection failed")

        with self.assertRaises(RuntimeError) as context:
            llm.chat([{"role": "user", "content": "test"}])

        self.assertIn("Ollama chat failed", str(context.exception))

        # 测试generate方法错误处理
        mock_ollama.generate.side_effect = Exception("Generation failed")

        with self.assertRaises(RuntimeError) as context:
            llm.generate("test prompt")

        self.assertIn("Ollama generate failed", str(context.exception))


class TestOpenAILLM(unittest.TestCase):
    """测试OpenAILLM类"""

    def setUp(self):
        """设置测试环境"""
        self.model_name = "gpt-3.5-turbo"
        self.api_key = "test_api_key"
        self.base_url = "https://api.openai.com/v1"

    @patch('llm.openai_llm.OpenAI')
    def test_openai_llm_initialization(self, mock_openai_class):
        """测试OpenAI LLM初始化"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        llm = OpenAILLM(self.model_name, self.api_key, self.base_url)

        self.assertEqual(llm.get_model_name(), self.model_name)
        self.assertFalse(llm.measure_usage)

        # 验证OpenAI客户端创建
        mock_openai_class.assert_called_once_with(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # 测试带使用统计的初始化
        llm_with_measure = OpenAILLM(
            self.model_name, self.api_key, measure_usage=True
        )
        self.assertTrue(llm_with_measure.measure_usage)
        self.assertEqual(llm_with_measure.total_tokens, 0)
        self.assertEqual(llm_with_measure.total_time, 0)

    @patch('llm.openai_llm.OpenAI')
    @patch('llm.openai_llm.time')
    def test_chat_method(self, mock_time, mock_openai_class):
        """测试chat方法"""
        # 设置时间mock
        mock_time.time.side_effect = [0.0, 1.0]  # 开始和结束时间

        # 设置OpenAI客户端mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 100

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = OpenAILLM(self.model_name, self.api_key, measure_usage=True)
        messages = [{"role": "user", "content": "Hello"}]

        response = llm.chat(messages, temperature=0.7)

        self.assertEqual(response, "Test response")

        # 验证使用统计
        self.assertEqual(llm.total_tokens, 100)
        self.assertEqual(llm.total_time, 1.0)
        self.assertEqual(llm.request_count, 1)

        # 验证API调用
        mock_client.chat.completions.create.assert_called_once_with(
            model=self.model_name,
            messages=messages,
            temperature=0.7
        )

    @patch('llm.openai_llm.OpenAI')
    def test_generate_method(self, mock_openai_class):
        """测试generate方法"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated text"

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = OpenAILLM(self.model_name, self.api_key)
        prompt = "Generate some text"

        response = llm.generate(prompt)

        self.assertEqual(response, "Generated text")

        # 验证转换为chat格式
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['messages'][0]['role'], 'user')
        self.assertEqual(call_args[1]['messages'][0]['content'], prompt)

    @patch('llm.openai_llm.OpenAI')
    def test_usage_stats(self, mock_openai_class):
        """测试使用统计功能"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        llm = OpenAILLM(self.model_name, self.api_key, measure_usage=True)

        # 初始状态
        stats = llm.get_usage_stats()
        self.assertEqual(stats['total_tokens'], 0)
        self.assertEqual(stats['request_count'], 0)

        # 手动设置一些统计数据
        llm.total_tokens = 200
        llm.total_time = 2.0
        llm.request_count = 2

        stats = llm.get_usage_stats()
        self.assertEqual(stats['total_tokens'], 200)
        self.assertEqual(stats['avg_tokens_per_request'], 100.0)
        self.assertEqual(stats['avg_time_per_request'], 1.0)

        # 测试重置统计
        llm.reset_usage_stats()
        self.assertEqual(llm.total_tokens, 0)
        self.assertEqual(llm.total_time, 0)
        self.assertEqual(llm.request_count, 0)

    @patch('llm.openai_llm.OpenAI')
    def test_error_handling(self, mock_openai_class):
        """测试错误处理"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        llm = OpenAILLM(self.model_name, self.api_key)

        with self.assertRaises(RuntimeError) as context:
            llm.chat([{"role": "user", "content": "test"}])

        self.assertIn("OpenAI chat failed", str(context.exception))


class TestLLMFactory(unittest.TestCase):
    """测试LLMFactory类"""

    def test_supported_providers(self):
        """测试支持的提供商"""
        providers = LLMFactory.list_supported_providers()

        self.assertIn('ollama', providers)
        self.assertIn('openai', providers)
        self.assertIsInstance(providers, list)

    def test_common_models(self):
        """测试常用模型列表"""
        models = LLMFactory.list_common_models()

        self.assertIsInstance(models, list)
        self.assertIn('llama3.1:8b', models)
        self.assertIn('gpt-3.5-turbo', models)

    def test_get_model_info(self):
        """测试获取模型信息"""
        info = LLMFactory.get_model_info('llama3.1:8b')
        self.assertEqual(info['provider'], 'ollama')

        info = LLMFactory.get_model_info('gpt-3.5-turbo')
        self.assertEqual(info['provider'], 'openai')

        # 测试不存在的模型
        info = LLMFactory.get_model_info('nonexistent_model')
        self.assertIsNone(info)

    @patch('llm.llm_factory.OllamaLLM')
    def test_create_ollama_llm(self, mock_ollama_class):
        """测试创建Ollama LLM"""
        mock_instance = Mock()
        mock_ollama_class.return_value = mock_instance

        llm = LLMFactory.create_ollama_llm("llama3.1:8b", host="localhost")

        mock_ollama_class.assert_called_once_with(
            model_name="llama3.1:8b", host="localhost"
        )
        self.assertEqual(llm, mock_instance)

    @patch('llm.llm_factory.OpenAILLM')
    def test_create_openai_llm(self, mock_openai_class):
        """测试创建OpenAI LLM"""
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        llm = LLMFactory.create_openai_llm(
            "gpt-3.5-turbo",
            api_key="test_key",
            base_url="https://api.openai.com/v1"
        )

        mock_openai_class.assert_called_once_with(
            model_name="gpt-3.5-turbo",
            api_key="test_key",
            base_url="https://api.openai.com/v1"
        )
        self.assertEqual(llm, mock_instance)

    def test_create_llm_unsupported_provider(self):
        """测试创建不支持的提供商LLM"""
        with self.assertRaises(ValueError) as context:
            LLMFactory.create_llm('unsupported_provider', 'model_name')

        self.assertIn('Unsupported provider', str(context.exception))

    @patch('llm.llm_factory.OllamaLLM')
    def test_from_config(self, mock_ollama_class):
        """测试从配置创建LLM"""
        mock_instance = Mock()
        mock_ollama_class.return_value = mock_instance

        config = {
            'provider': 'ollama',
            'model_name': 'llama3.1:8b',
            'host': 'localhost'
        }

        llm = LLMFactory.from_config(config)

        mock_ollama_class.assert_called_once_with(
            model_name='llama3.1:8b',
            host='localhost'
        )
        self.assertEqual(llm, mock_instance)

    @patch('llm.llm_factory.OpenAILLM')
    @patch('llm.llm_factory.OllamaLLM')
    def test_create_llm_generic(self, mock_ollama_class, mock_openai_class):
        """测试通用create_llm方法"""
        mock_ollama_instance = Mock()
        mock_openai_instance = Mock()
        mock_ollama_class.return_value = mock_ollama_instance
        mock_openai_class.return_value = mock_openai_instance

        # 测试Ollama
        ollama_llm = LLMFactory.create_llm('ollama', 'test_model')
        mock_ollama_class.assert_called_with(model_name='test_model')
        self.assertEqual(ollama_llm, mock_ollama_instance)

        # 测试OpenAI
        openai_llm = LLMFactory.create_llm('openai', 'gpt-3.5-turbo', api_key='test')
        mock_openai_class.assert_called_with(model_name='gpt-3.5-turbo', api_key='test')
        self.assertEqual(openai_llm, mock_openai_instance)


if __name__ == '__main__':
    print("=== 运行llm模块测试 ===")
    unittest.main(verbosity=2)
