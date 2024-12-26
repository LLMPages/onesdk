import os
import sys
import unittest
from typing import List, Dict
import time
from unittest.mock import patch, Mock

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onesdk.core import OneSDK
from onesdk.utils.error_handler import InvokeError, InvokeConnectionError, InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError
from onesdk.utils.logger import Logger, logger

class TestAnthropicAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestAnthropicAPI class")

        cls.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not cls.api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

        cls.base_url = os.environ.get("ANTHROPIC_URL")
        cls.sdk = OneSDK("anthropic", {
            "api_key": cls.api_key,
            "api_url": cls.base_url
        })
        cls.sdk.set_debug_mode(True)

        cls.default_model = "claude-3-5-sonnet-20241022"

    def setUp(self):
        time.sleep(1)  # 添加延迟以避免频率限制

    def test_list_models(self):
        logger.info("\nTesting list_models for Anthropic:")
        models = self.sdk.list_models()
        self.assertIsInstance(models, Dict)
        self.assertTrue(len(models) > 0)
        logger.info(f"Anthropic models: {models}")

    def test_base_url(self):
        logger.info("\nTesting base URL for Anthropic:")
        self.assertEqual(self.sdk.api.base_url, self.base_url)
        logger.info(f"Base URL correctly set to: {self.base_url}")

    def test_get_model(self):
        logger.info("\nTesting get_model for Anthropic:")
        model_info = self.sdk.get_model(self.default_model)
        self.assertIsInstance(model_info, Dict)
        self.assertEqual(model_info['id'], self.default_model)
        logger.info(f"Anthropic model info: {model_info}")

    def test_generate(self):
        logger.info("\nTesting generate for Anthropic:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        response = self.sdk.generate(self.default_model, messages, max_tokens=100)
        self.assertIsInstance(response, Dict)
        self.assertIn('content', response)
        logger.info(f"Anthropic response: {response['content']}")

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for Anthropic:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        stream = self.sdk.stream_generate(model=self.default_model, messages=messages, max_tokens=100)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for Anthropic")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            logger.info(f"Received chunk {chunk_count}: {chunk}")
            content = chunk.get('delta', {}).get('text', '')
            if content:
                full_response += content
                logger.info(f"Content: {content}")
        logger.info(f"\nAnthropic full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        self.assertTrue(chunk_count > 0, "No chunks were received")
        self.assertTrue(len(full_response) > 0, "No content was received")

    def test_count_tokens(self):
        logger.info("\nTesting count_tokens for Anthropic:")
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking. How can I assist you today?"}
        ]
        token_count = self.sdk.count_tokens(self.default_model, messages)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)
        logger.info(f"Anthropic token count: {token_count}")

    @patch.object(OneSDK, 'generate')
    def test_error_handling_generate(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])

    @patch.object(OneSDK, 'generate')
    def test_connection_error(self, mock_generate):
        mock_generate.side_effect = InvokeConnectionError("Connection error")
        with self.assertRaises(InvokeConnectionError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])

    @patch.object(OneSDK, 'generate')
    def test_rate_limit_error(self, mock_generate):
        mock_generate.side_effect = InvokeRateLimitError("Rate limit exceeded")
        with self.assertRaises(InvokeRateLimitError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])

    @patch.object(OneSDK, 'generate')
    def test_authorization_error(self, mock_generate):
        mock_generate.side_effect = InvokeAuthorizationError("Invalid API key")
        with self.assertRaises(InvokeAuthorizationError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])

    # def test_set_proxy(self):
    #     logger.info("\nTesting set_proxy for Anthropic:")
    #     proxy_url = "http://example.com:8080"  # 使用一个示例 URL，不要实际连接
    #     self.sdk.set_proxy(proxy_url)
    #     logger.info(f"Proxy set to {proxy_url}")
    #     # 注意：这里我们只是测试方法调用，不测试实际连接

if __name__ == "__main__":
    unittest.main(verbosity=2)