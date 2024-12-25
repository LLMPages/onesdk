import os
import sys
import unittest
from typing import List, Dict
import time
import asyncio
from unittest.mock import patch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core import OneSDK
from ..utils.error_handler import InvokeError

class TestOneSDKUser(unittest.TestCase):
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

        cls.providers = {
            "anthropic": OneSDK("anthropic", {"api_key": cls.api_keys["anthropic"]}),
            "qwen": OneSDK("qwen", {"api_key": cls.api_keys["qwen"]}),
            "cohere": OneSDK("cohere", {"api_key": cls.api_keys["cohere"]}),
            "doubao": OneSDK("doubao", {"api_key": cls.api_keys["doubao"]}),
            "gemini": OneSDK("gemini", {"api_key": cls.api_keys["gemini"]}),
            "minimax": OneSDK("minimax", {"api_key": cls.api_keys["minimax"], "group_id": cls.api_keys["minimax_group_id"]}),
            "openai": OneSDK("openai", {"api_key": cls.api_keys["openai"]}),
            "wenxin": OneSDK("wenxin", {"api_key": cls.api_keys["wenxin"], "secret_key": cls.api_keys["wenxin_secret"]})
        }

        cls.default_models = {
            "anthropic": "claude-3-opus-20240229",
            "qwen": "qwen-turbo",
            "cohere": "command",
            "doubao": "doubao-v1",
            "gemini": "gemini-pro",
            "minimax": "abab5-chat",
            "openai": "gpt-3.5-turbo",
            "wenxin": "ERNIE-Bot"
        }

    def test_list_models(self):
        print("\nTesting list_models for all providers:")
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    models = sdk.list_models()
                    self.assertIsInstance(models, List)
                    self.assertTrue(len(models) > 0)
                    print(f"{provider_name.capitalize()} models: {models}")
                except Exception as e:
                    self.fail(f"Error occurred while listing models for {provider_name}: {str(e)}")

    def test_generate(self):
        print("\nTesting generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    response = sdk.generate(model, messages)
                    self.assertIsInstance(response, Dict)
                    self.assertIn('choices', response)
                    self.assertIn('message', response['choices'][0])
                    print(f"{provider_name.capitalize()} response: {response['choices'][0]['message']['content']}")
                except Exception as e:
                    self.fail(f"Error occurred during generate for {provider_name}: {str(e)}")

    def test_stream_generate(self):
        print("\nTesting stream_generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    stream = sdk.stream_generate(model=model, messages=messages)
                    full_response = ""
                    chunk_count = 0
                    start_time = time.time()
                    timeout = 30  # 30 seconds timeout
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
                    print(f"\n{provider_name.capitalize()} full response: {full_response}")
                    print(f"Total chunks received: {chunk_count}")
                    print(f"Time taken: {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    self.fail(f"Error occurred during stream_generate for {provider_name}: {str(e)}")

    def test_count_tokens(self):
        print("\nTesting count_tokens for all providers:")
        messages = [{"role": "user", "content": "Hello, world!"}]
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    token_count = sdk.count_tokens(model, messages)
                    self.assertIsInstance(token_count, int)
                    self.assertTrue(token_count > 0)
                    print(f"{provider_name.capitalize()} token count: {token_count}")
                except Exception as e:
                    self.fail(f"Error occurred during count_tokens for {provider_name}: {str(e)}")

    @patch.object(OneSDK, 'generate')
    def test_error_handling(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.providers['anthropic'].generate(self.default_models['anthropic'], [{"role": "user", "content": "Test"}])

    def test_set_model(self):
        print("\nTesting set_model for all providers:")
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    sdk_with_model = sdk.set_model(model)
                    self.assertEqual(sdk_with_model.current_model, model)
                    print(f"{provider_name.capitalize()} model set successfully: {model}")
                except Exception as e:
                    self.fail(f"Error occurred during set_model for {provider_name}: {str(e)}")

    @unittest.skip("Async test - run separately")
    def test_async_generate(self):
        print("\nTesting async_generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]

        async def run_async_generate():
            for provider_name, sdk in self.providers.items():
                with self.subTest(provider=provider_name):
                    try:
                        model = self.default_models[provider_name]
                        response = await sdk.async_generate(model, messages)
                        self.assertIsInstance(response, Dict)
                        self.assertIn('choices', response)
                        self.assertIn('message', response['choices'][0])
                        print(f"{provider_name.capitalize()} async response: {response['choices'][0]['message']['content']}")
                    except Exception as e:
                        self.fail(f"Error occurred during async_generate for {provider_name}: {str(e)}")

        asyncio.run(run_async_generate())

if __name__ == "__main__":
    unittest.main()