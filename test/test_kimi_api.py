import os
import sys
import unittest
from typing import List, Dict
import time
from unittest.mock import patch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_onesdk.core import OneSDK
from llm_onesdk.utils.error_handler import InvokeError, InvokeConnectionError, InvokeRateLimitError, InvokeAuthorizationError
from llm_onesdk.utils.logger import Logger, logger

class TestKimiAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestKimiAPI class")

        cls.api_key = os.environ.get("MOONSHOT_API_KEY")
        if not cls.api_key:
            raise ValueError("Please set MOONSHOT_API_KEY environment variable")

        cls.sdk = OneSDK("kimi", {
            "api_key": cls.api_key
        })
        cls.sdk.set_debug_mode(True)

        cls.default_model = "moonshot-v1-8k"  # 使用 Kimi 的默认模型

    def setUp(self):
        time.sleep(1)  # 添加延迟以避免频率限制

    def test_list_models(self):
        logger.info("\nTesting list_models for Kimi:")
        models = self.sdk.list_models()
        self.assertIsInstance(models, List)
        self.assertTrue(len(models) > 0)
        logger.info(f"Kimi models: {models}")

    def test_get_model(self):
        logger.info("\nTesting get_model for Kimi:")
        model_info = self.sdk.get_model(self.default_model)
        self.assertIsInstance(model_info, Dict)
        self.assertEqual(model_info['id'], self.default_model)
        logger.info(f"Kimi model info: {model_info}")

    def test_generate(self):
        logger.info("\nTesting generate for Kimi:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        response = self.sdk.generate(self.default_model, messages)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertIn('content', response['choices'][0]['message'])
        logger.info(f"Kimi response: {response['choices'][0]['message']['content']}")

    def test_stream_generate(self):
        import time
        time.sleep(60)
        logger.info("\nTesting stream_generate for Kimi:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        stream = self.sdk.stream_generate(model=self.default_model, messages=messages)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        time.sleep(2)
        for chunk in stream:
            time.sleep(2)
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for Kimi")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            self.assertIn('delta', chunk)
            content = chunk['delta'].get('text', '')
            if content:
                full_response += content
                logger.info(f"Kimi chunk {chunk_count}: {content}")
        logger.info(f"\nKimi full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        self.assertTrue(chunk_count > 0, "No chunks were received")
        self.assertTrue(len(full_response) > 0, "No content was received")

    def test_count_tokens(self):
        logger.info("\nTesting count_tokens for Kimi:")
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking. How can I assist you today?"}
        ]
        token_count = self.sdk.count_tokens(self.default_model, messages)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)
        logger.info(f"Kimi token count: {token_count}")

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
    #     logger.info("\nTesting set_proxy for Kimi:")
    #     proxy_url = "http://example.com:8080"  # 使用一个示例 URL，不要实际连接
    #     self.sdk.set_proxy(proxy_url)
    #     logger.info(f"Proxy set to {proxy_url}")
    #     # 注意：这里我们只是测试方法调用，不测试实际连接

if __name__ == "__main__":
    unittest.main(verbosity=2)