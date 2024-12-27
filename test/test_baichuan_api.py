import os
import sys
import unittest
from typing import Dict
import time
from unittest.mock import patch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_onesdk.core import OneSDK
from llm_onesdk.utils.error_handler import InvokeError, InvokeBadRequestError, InvokeRateLimitError
from llm_onesdk.utils.logger import Logger, logger

class TestBaichuanAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestBaichuanAPI class")

        cls.api_key = os.environ.get("BAICHUAN_API_KEY")
        if not cls.api_key:
            raise ValueError("Please set BAICHUAN_API_KEY environment variable")

        cls.sdk = OneSDK("baichuan", {"api_key": cls.api_key})
        cls.sdk.set_debug_mode(True)

        cls.chat_model = "Baichuan2-53B"
        cls.embedding_model = "Baichuan-Text-Embedding"

    def setUp(self):
        time.sleep(1)  # 添加延迟以避免频率限制

    def test_generate(self):
        logger.info("\nTesting generate for Baichuan:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        response = self.sdk.generate(self.chat_model, messages)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertIn('message', response['choices'][0])
        logger.info(f"Baichuan response: {response['choices'][0]['message']['content']}")

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for Baichuan:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        stream = self.sdk.stream_generate(model=self.chat_model, messages=messages)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for Baichuan")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            self.assertIn('choices', chunk)
            self.assertIn('delta', chunk['choices'][0])
            content = chunk['choices'][0]['delta'].get('content', '')
            if content:
                full_response += content
                logger.info(f"Baichuan chunk {chunk_count}: {content}")
        logger.info(f"\nBaichuan full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

    def test_create_embedding_single(self):
        logger.info("\nTesting create_embedding (single input) for Baichuan:")
        input_text = "百川大模型"
        response = self.sdk.create_embedding(self.embedding_model, input_text)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertEqual(len(response['data']), 1)
        self.assertIn('embedding', response['data'][0])
        self.assertIn('usage', response)
        logger.info(f"Baichuan embedding: {response['data'][0]['embedding'][:5]}...")
        logger.info(f"Usage: {response['usage']}")

    def test_create_embedding_batch(self):
        logger.info("\nTesting create_embedding (batch input) for Baichuan:")
        input_texts = ["新年快乐", "百川大模型"]
        response = self.sdk.create_embedding(self.embedding_model, input_texts)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertEqual(len(response['data']), 2)
        for item in response['data']:
            self.assertIn('embedding', item)
        self.assertIn('usage', response)
        logger.info(f"Baichuan embeddings: {[item['embedding'][:5] for item in response['data']]}")
        logger.info(f"Usage: {response['usage']}")

    @patch.object(OneSDK, 'generate')
    def test_error_handling_generate(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.sdk.generate(self.chat_model, [{"role": "user", "content": "Test"}])

    @patch.object(OneSDK, 'create_embedding')
    def test_error_handling_embedding(self, mock_create_embedding):
        mock_create_embedding.side_effect = InvokeBadRequestError("Invalid input")
        with self.assertRaises(InvokeBadRequestError):
            self.sdk.create_embedding(self.embedding_model, "")  # 空输入应该触发错误

    @patch.object(OneSDK, 'create_embedding')
    def test_rate_limit_embedding(self, mock_create_embedding):
        mock_create_embedding.side_effect = InvokeRateLimitError("Rate limit exceeded")
        with self.assertRaises(InvokeRateLimitError):
            self.sdk.create_embedding(self.embedding_model, "Test rate limit")

if __name__ == "__main__":
    unittest.main(verbosity=2)