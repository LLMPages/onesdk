import os
import sys
import unittest
from typing import Dict
import time
from unittest.mock import patch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_onesdk.core import OneSDK
from llm_onesdk.utils.error_handler import InvokeError
from llm_onesdk.utils.logger import Logger, logger

class TestQwenAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestQwenAPI class")

        cls.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not cls.api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")

        cls.sdk = OneSDK("qwen", {"api_key": cls.api_key})
        cls.sdk.set_debug_mode(True)

        cls.default_model = "qwen-turbo"

    def test_count_tokens(self):
        logger.info("\nTesting count_tokens for Qwen:")
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking. How can I assist you today?"}
        ]
        token_count = self.sdk.count_tokens(self.default_model, messages)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)
        logger.info(f"Qwen token count: {token_count}")

    def test_generate(self):
        logger.info("\nTesting generate for Wenxin:")
        messages = [{"role": "user", "content": "请从1数到5。"}]
        response = self.sdk.generate(self.default_model, messages)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertTrue(len(response['choices']) > 0, "Choices list is empty")
        self.assertIn('message', response['choices'][0])
        content = response['choices'][0]['message'].get('content', '')
        logger.info(f"Wenxin response: {content}")

        # 添加更多的断言来检查响应的内容
        self.assertTrue(content, "Response content is empty")
        if "Error" in content:
            logger.warning(f"API returned an error response: {content}")
        else:
            self.assertIn("1", content)
            self.assertIn("5", content)

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for Qwen:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        stream = self.sdk.stream_generate(model=self.default_model, messages=messages)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for Qwen")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            self.assertIn('choices', chunk)
            self.assertIn('message', chunk['choices'][0])
            content = chunk['choices'][0]['message'].get('content', '')
            if content:
                full_response += content
                logger.info(f"Qwen chunk {chunk_count}: {content}")
        logger.info(f"\nQwen full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

    def test_set_model(self):
        logger.info("\nTesting set_model for Qwen:")
        sdk_with_model = self.sdk.set_model(self.default_model)
        self.assertEqual(sdk_with_model.current_model, self.default_model)
        logger.info(f"Qwen model set successfully: {self.default_model}")

    @patch.object(OneSDK, 'generate')
    def test_error_handling(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])

if __name__ == "__main__":
    unittest.main(verbosity=2)