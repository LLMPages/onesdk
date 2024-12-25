import os
import sys
import unittest
from typing import List, Dict
import time
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core import OneSDK

class TestOneSDKUser(unittest.TestCase):
    def setUp(self):
        # 设置 API 密钥（请确保在环境变量中设置了这些值）
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.aliyun_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.cohere_api_key = os.environ.get("COHERE_API_KEY")
        self.doubao_api_key = os.environ.get("DOUBAO_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.minimax_api_key = os.environ.get("MINIMAX_API_KEY")
        self.minimax_group_id = os.environ.get("MINIMAX_GROUP_ID")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.wenxin_api_key = os.environ.get("WENXIN_API_KEY")
        self.wenxin_secret_key = os.environ.get("WENXIN_SECRET_KEY")

        if not all([self.anthropic_api_key, self.aliyun_api_key, self.cohere_api_key, 
                    self.doubao_api_key, self.gemini_api_key, self.minimax_api_key, 
                    self.minimax_group_id, self.openai_api_key, self.wenxin_api_key, 
                    self.wenxin_secret_key]):
            raise ValueError("Please set all required API keys in environment variables")

        self.providers = {
            "anthropic": OneSDK("anthropic", {"api_key": self.anthropic_api_key}),
            "qwen": OneSDK("qwen", {"api_key": self.aliyun_api_key}),
            "cohere": OneSDK("cohere", {"api_key": self.cohere_api_key}),
            "doubao": OneSDK("doubao", {"api_key": self.doubao_api_key}),
            "gemini": OneSDK("gemini", {"api_key": self.gemini_api_key}),
            "minimax": OneSDK("minimax", {"api_key": self.minimax_api_key, "group_id": self.minimax_group_id}),
            "openai": OneSDK("openai", {"api_key": self.openai_api_key}),
            "wenxin": OneSDK("wenxin", {"api_key": self.wenxin_api_key, "secret_key": self.wenxin_secret_key})
        }

    def test_list_models(self):
        print("\nTesting list_models for all providers:")
        for provider_name, sdk in self.providers.items():
            models = sdk.list_models()
            self.assertIsInstance(models, List)
            self.assertTrue(len(models) > 0)
            print(f"{provider_name.capitalize()} models: {models}")

    def test_generate(self):
        print("\nTesting generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for provider_name, sdk in self.providers.items():
            model = self._get_default_model(provider_name)
            response = sdk.generate(model, messages)
            self.assertIsInstance(response, Dict)
            self.assertIn('choices', response)
            self.assertIn('message', response['choices'][0])
            print(f"{provider_name.capitalize()} response: {response['choices'][0]['message']['content']}")

    def test_stream_generate(self):
        print("\nTesting stream_generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for provider_name, sdk in self.providers.items():
            model = self._get_default_model(provider_name)
            stream = sdk.stream_generate(model=model, messages=messages)
            full_response = ""
            chunk_count = 0
            start_time = time.time()
            timeout = 30  # 30 seconds timeout
            try:
                for chunk in stream:
                    if time.time() - start_time > timeout:
                        print(f"Timeout reached for {provider_name}")
                        break
                    chunk_count += 1
                    self.assertIsInstance(chunk, Dict)
                    self.assertIn('choices', chunk)
                    self.assertIn('message', chunk['choices'][0])
                    content = chunk['choices'][0]['message'].get('content', '')
                    if content:
                        full_response += content
                        print(f"{provider_name.capitalize()} chunk {chunk_count}: {content}", flush=True)
            except Exception as e:
                print(f"Error occurred during streaming for {provider_name}: {str(e)}")

            print(f"\n{provider_name.capitalize()} full response: {full_response}")
            print(f"Total chunks received: {chunk_count}")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")

    def test_count_tokens(self):
        print("\nTesting count_tokens for all providers:")
        messages = [{"role": "user", "content": "Hello, world!"}]
        for provider_name, sdk in self.providers.items():
            model = self._get_default_model(provider_name)
            token_count = sdk.count_tokens(model, messages)
            self.assertIsInstance(token_count, int)
            self.assertTrue(token_count > 0)
            print(f"{provider_name.capitalize()} token count: {token_count}")

    def _get_default_model(self, provider_name):
        default_models = {
            "anthropic": "claude-3-opus-20240229",
            "qwen": "qwen-turbo",
            "cohere": "command",
            "doubao": "doubao-v1",
            "gemini": "gemini-pro",
            "minimax": "abab5-chat",
            "openai": "gpt-3.5-turbo",
            "wenxin": "ERNIE-Bot"
        }
        return default_models.get(provider_name, "")

if __name__ == "__main__":
    unittest.main()