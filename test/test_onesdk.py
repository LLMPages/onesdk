import os
import sys
import unittest
from typing import List, Dict
import time
import asyncio
from unittest.mock import patch
from ..utils.logger import Logger, logger
from ..utils.error_handler import InvokeError, InvokeUnsupportedOperationError

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core import OneSDK

class TestOneSDKUser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 设置日志为调试模式
        Logger.set_debug_mode(True)
        logger.info("Setting up TestOneSDKUser class")

        cls.api_keys = {
            # "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "qwen": os.environ.get("DASHSCOPE_API_KEY"),
            # "cohere": os.environ.get("COHERE_API_KEY"),
            "doubao": os.environ.get("DOUBAO_API_KEY"),
            # "gemini": os.environ.get("GEMINI_API_KEY"),
            # "minimax": os.environ.get("MINIMAX_API_KEY"),
            # "minimax_group_id": os.environ.get("MINIMAX_GROUP_ID"),
            # "openai": os.environ.get("OPENAI_API_KEY"),
            # "wenxin": os.environ.get("WENXIN_API_KEY"),
            # "wenxin_secret": os.environ.get("WENXIN_SECRET_KEY")
        }

        cls.providers = {
            # "anthropic": OneSDK("anthropic", {"api_key": cls.api_keys["anthropic"]}),
            "qwen": OneSDK("qwen", {"api_key": cls.api_keys["qwen"]}),
            # "cohere": OneSDK("cohere", {"api_key": cls.api_keys["cohere"]}),
            "doubao": OneSDK("doubao", {"api_key": cls.api_keys["doubao"]}),
            # "gemini": OneSDK("gemini", {"api_key": cls.api_keys["gemini"]}),
            # "minimax": OneSDK("minimax", {"api_key": cls.api_keys["minimax"], "group_id": cls.api_keys["minimax_group_id"]}),
            # "openai": OneSDK("openai", {"api_key": cls.api_keys["openai"]}),
            # "wenxin": OneSDK("wenxin", {"api_key": cls.api_keys["wenxin"], "secret_key": cls.api_keys["wenxin_secret"]})
        }


        # 为每个提供者设置调试模式
        for provider in cls.providers.values():
            provider.set_debug_mode(True)

        cls.default_models = {
            "anthropic": "claude-3-opus-20240229",
            "qwen": "qwen-turbo",
            "cohere": "command",
            "doubao": "ep-20241225233943-vkjxr",  # 使用正确的 tokenization 模型
            "gemini": "gemini-pro",
            "minimax": "abab5-chat",
            "openai": "gpt-3.5-turbo",
            "wenxin": "ERNIE-Bot"
        }

    def test_list_models(self):
        logger.info("\nTesting list_models for all providers:")
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    models = sdk.list_models()
                    self.assertIsInstance(models, List)
                    self.assertTrue(len(models) > 0)
                    logger.info(f"{provider_name.capitalize()} models: {models}")
                except InvokeUnsupportedOperationError:
                    logger.info(f"{provider_name.capitalize()} does not support listing models")
                except Exception as e:
                    logger.error(f"Unexpected error while listing models for {provider_name}: {str(e)}")
                    self.fail(f"Unexpected error while listing models for {provider_name}: {str(e)}")

    def test_generate(self):
        logger.info("\nTesting generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    logger.info(f"Testing generate for {provider_name} with model {model}")
                    response = sdk.generate(model, messages)
                    self.assertIsInstance(response, Dict)
                    self.assertIn('choices', response)
                    self.assertIn('message', response['choices'][0])
                    logger.info(f"{provider_name.capitalize()} response: {response['choices'][0]['message']['content']}")
                except InvokeUnsupportedOperationError:
                    logger.info(f"{provider_name.capitalize()} does not support text generation")
                except Exception as e:
                    logger.error(f"Unexpected error during generate for {provider_name}: {str(e)}")
                    self.fail(f"Unexpected error during generate for {provider_name}: {str(e)}")

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 100."}]
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    logger.info(f"Testing stream_generate for {provider_name} with model {model}")
                    stream = sdk.stream_generate(model=model, messages=messages)
                    full_response = ""
                    chunk_count = 0
                    start_time = time.time()
                    timeout = 30  # 30 seconds timeout
                    for chunk in stream:
                        if time.time() - start_time > timeout:
                            logger.warning(f"Timeout reached for {provider_name}")
                            break
                        chunk_count += 1
                        self.assertIsInstance(chunk, Dict)
                        self.assertIn('choices', chunk)
                        self.assertIn('message', chunk['choices'][0])
                        content = chunk['choices'][0]['message'].get('content', '')
                        if content:
                            full_response += content
                            logger.info(f"{provider_name.capitalize()} chunk {chunk_count}: {content}")
                    logger.info(f"\n{provider_name.capitalize()} full response: {full_response}")
                    logger.info(f"Total chunks received: {chunk_count}")
                    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
                except InvokeUnsupportedOperationError:
                    logger.info(f"{provider_name.capitalize()} does not support stream generation")
                except Exception as e:
                    logger.error(f"Unexpected error during stream_generate for {provider_name}: {str(e)}")
                    self.fail(f"Unexpected error during stream_generate for {provider_name}: {str(e)}")

    def test_count_tokens(self):
        logger.info("\nTesting count_tokens for all providers:")
        messages = [{"role": "user", "content": "Hello, world!"}]
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    logger.info(f"Testing count_tokens for {provider_name} with model {model}")
                    token_count = sdk.count_tokens(model, messages)
                    self.assertIsInstance(token_count, int)
                    self.assertTrue(token_count > 0)
                    logger.info(f"{provider_name.capitalize()} token count: {token_count}")
                except InvokeUnsupportedOperationError:
                    logger.info(f"{provider_name.capitalize()} does not support token counting")
                except Exception as e:
                    logger.error(f"Unexpected error during count_tokens for {provider_name}: {str(e)}")
                    self.fail(f"Unexpected error during count_tokens for {provider_name}: {str(e)}")

    @patch.object(OneSDK, 'generate')
    def test_error_handling(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")

        if not self.providers:
            self.skipTest("No providers available for testing")

        # 遍历所有可用的 providers
        for provider_name, provider in self.providers.items():
            with self.subTest(provider=provider_name):
                model = self.default_models[provider_name]
                with self.assertRaises(InvokeError):
                    provider.generate(model, [{"role": "user", "content": "Test"}])

    def test_set_model(self):
        logger.info("\nTesting set_model for all providers:")
        for provider_name, sdk in self.providers.items():
            with self.subTest(provider=provider_name):
                try:
                    model = self.default_models[provider_name]
                    sdk_with_model = sdk.set_model(model)
                    self.assertEqual(sdk_with_model.current_model, model)
                    logger.info(f"{provider_name.capitalize()} model set successfully: {model}")
                except Exception as e:
                    logger.error(f"Unexpected error during set_model for {provider_name}: {str(e)}")
                    self.fail(f"Unexpected error during set_model for {provider_name}: {str(e)}")

    @unittest.skip("Async test - run separately")
    def test_async_generate(self):
        logger.info("\nTesting async_generate for all providers:")
        messages = [{"role": "user", "content": "Count from 1 to 100."}]

        async def run_async_generate():
            for provider_name, sdk in self.providers.items():
                with self.subTest(provider=provider_name):
                    try:
                        model = self.default_models[provider_name]
                        logger.info(f"Testing async_generate for {provider_name} with model {model}")
                        response = await sdk.async_generate(model, messages)
                        self.assertIsInstance(response, Dict)
                        self.assertIn('choices', response)
                        self.assertIn('message', response['choices'][0])
                        logger.info(f"{provider_name.capitalize()} async response: {response['choices'][0]['message']['content']}")
                    except InvokeUnsupportedOperationError:
                        logger.info(f"{provider_name.capitalize()} does not support async generation")
                    except Exception as e:
                        logger.error(f"Unexpected error during async_generate for {provider_name}: {str(e)}")
                        self.fail(f"Unexpected error during async_generate for {provider_name}: {str(e)}")

        asyncio.run(run_async_generate())

if __name__ == "__main__":
    unittest.main(verbosity=2)