import os
import sys
import unittest
from typing import List, Dict

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core import OneSDK

class TestOneSDKUser(unittest.TestCase):
    def setUp(self):
        # 设置 API 密钥（请确保在环境变量中设置了这些值）
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.aliyun_api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not self.anthropic_api_key or not self.aliyun_api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY and DASHSCOPE_API_KEY environment variables")

        self.providers = {
            "anthropic": OneSDK("anthropic", {"api_key": self.anthropic_api_key}),
            "aliyun": OneSDK("aliyun", {"api_key": self.aliyun_api_key})
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
        messages = [{"role": "user", "content": "What is the capital of France?"}]
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
            stream = sdk.stream_generate(model, messages)
            full_response = ""
            for chunk in stream:
                self.assertIsInstance(chunk, Dict)
                self.assertIn('choices', chunk)
                self.assertIn('message', chunk['choices'][0])
                content = chunk['choices'][0]['message'].get('content', '')
                if content:
                    full_response += content
                    print(f"{provider_name.capitalize()} chunk: {content}", end='', flush=True)
            print(f"\n{provider_name.capitalize()} full response: {full_response}")

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
        if provider_name == "anthropic":
            return "claude-3-opus-20240229"
        elif provider_name == "aliyun":
            return "qwen-turbo"
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

if __name__ == "__main__":
    unittest.main()