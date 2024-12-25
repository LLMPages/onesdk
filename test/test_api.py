import os
import sys
import unittest
from typing import List, Dict
from unittest.mock import patch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anthropic.api import API as AnthropicAPI
from models.qwen.api import API as QwenAPI
from models.cohere.api import API as CohereAPI
from models.doubao.api import API as DoubaoAPI
from models.gemini.api import API as GeminiAPI
from models.minimax.api import API as MinimaxAPI
from models.openai.api import API as OpenAIAPI
from models.wenxin.api import API as WenxinAPI
from models.ollama.api import API as OllamaAPI

from utils.error_handler import InvokeError

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_keys = {
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "qwen": os.environ.get("DASHSCOPE_API_KEY"),
            "cohere": os.environ.get("COHERE_API_KEY"),
            "doubao": os.environ.get("DOUBAO_API_KEY"),
            "gemini": os.environ.get("GEMINI_API_KEY"),
            "minimax": os.environ.get("MINIMAX_API_KEY"),
            "minimax_group_id": os.environ.get("MINIMAX_GROUP_ID"),
            "openai": os.environ.get("OPENAI_API_KEY"),
            "wenxin": os.environ.get("WENXIN_API_KEY"),
            "wenxin_secret": os.environ.get("WENXIN_SECRET_KEY")
        }

        if not all(cls.api_keys.values()):
            raise ValueError("Please set all required API keys in environment variables")

        cls.apis = {
            "anthropic": AnthropicAPI({"api_key": cls.api_keys["anthropic"]}),
            "qwen": QwenAPI({"api_key": cls.api_keys["qwen"]}),
            "cohere": CohereAPI({"api_key": cls.api_keys["cohere"]}),
            "doubao": DoubaoAPI({"api_key": cls.api_keys["doubao"]}),
            "gemini": GeminiAPI({"api_key": cls.api_keys["gemini"]}),
            "minimax": MinimaxAPI({"api_key": cls.api_keys["minimax"], "group_id": cls.api_keys["minimax_group_id"]}),
            "openai": OpenAIAPI({"api_key": cls.api_keys["openai"]}),
            "wenxin": WenxinAPI({"api_key": cls.api_keys["wenxin"], "secret_key": cls.api_keys["wenxin_secret"]}),
            "ollama": OllamaAPI({})  # Ollama doesn't require API keys
        }

        cls.default_models = {
            "anthropic": "claude-3-opus-20240229",
            "qwen": "qwen-turbo",
            "cohere": "command",
            "doubao": "doubao-v1",
            "gemini": "gemini-pro",
            "minimax": "abab5-chat",
            "openai": "gpt-3.5-turbo",
            "wenxin": "ERNIE-Bot",
            "ollama": "llama2"
        }

    def test_list_models(self):
        print("\nTesting list_models:")
        for name, api in self.apis.items():
            with self.subTest(api=name):
                try:
                    models = api.list_models()
                    self.assertIsInstance(models, List)
                    self.assertTrue(len(models) > 0)
                    print(f"{name.capitalize()} models: {models}")
                except NotImplementedError:
                    print(f"{name.capitalize()} does not support list_models")
                except Exception as e:
                    self.fail(f"Error occurred while listing models for {name}: {str(e)}")

    def test_generate(self):
        print("\nTesting generate:")
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        for name, api in self.apis.items():
            with self.subTest(api=name):
                try:
                    model = self.default_models[name]
                    response = api.generate(model, messages)
                    self.assertIsInstance(response, Dict)
                    self.assertIn('choices', response)
                    self.assertIn('message', response['choices'][0])
                    print(f"{name.capitalize()} response: {response['choices'][0]['message']['content']}")
                except Exception as e:
                    self.fail(f"Error occurred during generate for {name}: {str(e)}")

    def test_stream_generate(self):
        print("\nTesting stream_generate:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for name, api in self.apis.items():
            with self.subTest(api=name):
                try:
                    model = self.default_models[name]
                    stream = api.stream_generate(model, messages)
                    full_response = ""
                    for chunk in stream:
                        self.assertIsInstance(chunk, Dict)
                        self.assertIn('choices', chunk)
                        self.assertIn('message', chunk['choices'][0])
                        content = chunk['choices'][0]['message'].get('content', '')
                        if content:
                            full_response += content
                            print(f"{name.capitalize()} chunk: {content}", end='', flush=True)
                    print(f"\n{name.capitalize()} full response: {full_response}")
                except Exception as e:
                    self.fail(f"Error occurred during stream_generate for {name}: {str(e)}")

    def test_count_tokens(self):
        print("\nTesting count_tokens:")
        messages = [{"role": "user", "content": "Hello, world!"}]
        for name, api in self.apis.items():
            with self.subTest(api=name):
                try:
                    model = self.default_models[name]
                    token_count = api.count_tokens(model, messages)
                    self.assertIsInstance(token_count, int)
                    self.assertTrue(token_count > 0)
                    print(f"{name.capitalize()} token count: {token_count}")
                except Exception as e:
                    self.fail(f"Error occurred during count_tokens for {name}: {str(e)}")

    @patch.object(AnthropicAPI, 'generate')
    def test_error_handling(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.apis['anthropic'].generate(self.default_models['anthropic'], [{"role": "user", "content": "Test"}])

    def test_set_proxy(self):
        print("\nTesting set_proxy:")
        proxy_url = "http://test-proxy.com:8080"
        for name, api in self.apis.items():
            with self.subTest(api=name):
                try:
                    api.set_proxy(proxy_url)
                    self.assertEqual(api.session.proxies['http'], proxy_url)
                    self.assertEqual(api.session.proxies['https'], proxy_url)
                    print(f"{name.capitalize()} proxy set successfully")
                except Exception as e:
                    self.fail(f"Error occurred during set_proxy for {name}: {str(e)}")

if __name__ == "__main__":
    unittest.main()