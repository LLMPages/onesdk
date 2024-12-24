# test_api.py

import os
import sys
import unittest
from typing import List, Dict

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anthropic.api import API as AnthropicAPI
from models.aliyun.api import API as AliyunAPI


class TestAPI(unittest.TestCase):
    def setUp(self):
        # 设置 API 密钥（请确保在环境变量中设置了这些值）
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.aliyun_api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not self.anthropic_api_key or not self.aliyun_api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY and DASHSCOPE_API_KEY environment variables")

        self.anthropic_api = AnthropicAPI({"api_key": self.anthropic_api_key})
        self.aliyun_api = AliyunAPI({"api_key": self.aliyun_api_key})

    def test_list_models(self):
        print("\nTesting list_models:")
        for api in [self.anthropic_api, self.aliyun_api]:
            models = api.list_models()
            self.assertIsInstance(models, List)
            self.assertTrue(len(models) > 0)
            print(f"{api.__class__.__name__} models: {models}")

    def test_generate(self):
        print("\nTesting generate:")
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        for api in [self.anthropic_api, self.aliyun_api]:
            if isinstance(api, AnthropicAPI):
                model = "claude-3-opus-20240229"
            else:
                model = "qwen-turbo"

            response = api.generate(model, messages)
            self.assertIsInstance(response, Dict)
            self.assertIn('choices', response)
            self.assertIn('message', response['choices'][0])
            print(f"{api.__class__.__name__} response: {response['choices'][0]['message']['content']}")

    def test_stream_generate(self):
        print("\nTesting stream_generate:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for api in [self.anthropic_api, self.aliyun_api]:
            if isinstance(api, AnthropicAPI):
                model = "claude-3-opus-20240229"
            else:
                model = "qwen-turbo"

            stream = api.stream_generate(model, messages)
            full_response = ""
            for chunk in stream:
                self.assertIsInstance(chunk, Dict)
                self.assertIn('choices', chunk)
                self.assertIn('message', chunk['choices'][0])
                content = chunk['choices'][0]['message'].get('content', '')
                if content:
                    full_response += content
                    print(f"{api.__class__.__name__} chunk: {content}", end='', flush=True)
            print(f"\n{api.__class__.__name__} full response: {full_response}")

    def test_count_tokens(self):
        print("\nTesting count_tokens:")
        messages = [{"role": "user", "content": "Hello, world!"}]
        for api in [self.anthropic_api, self.aliyun_api]:
            if isinstance(api, AnthropicAPI):
                model = "claude-3-opus-20240229"
            else:
                model = "qwen-turbo"

            token_count = api.count_tokens(model, messages)
            self.assertIsInstance(token_count, int)
            self.assertTrue(token_count > 0)
            print(f"{api.__class__.__name__} token count: {token_count}")


if __name__ == "__main__":
    unittest.main()