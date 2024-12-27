import os
import sys
import unittest
from typing import List, Dict
import time
from unittest.mock import patch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_onesdk.core import OneSDK
from llm_onesdk.utils.error_handler import InvokeError, InvokeUnsupportedOperationError
from llm_onesdk.utils.logger import Logger, logger

class TestDoubaoAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestDoubaoAPI class")

        cls.api_key = os.environ.get("DOUBAO_API_KEY")
        if not cls.api_key:
            raise ValueError("Please set DOUBAO_API_KEY environment variable")

        cls.sdk = OneSDK("doubao", {"api_key": cls.api_key})
        cls.sdk.set_debug_mode(True)

        cls.default_model = "ep-20241225233943-vkjxr"  # 使用正确的 tokenization 模型

    def test_count_tokens(self):
        logger.info("\nTesting count_tokens for Doubao:")
        messages = [
            {"role": "user", "content": "天空为什么这么蓝"},
            {"role": "user", "content": "花儿为什么这么香"}
        ]
        try:
            token_count = self.sdk.count_tokens(self.default_model, messages)
            self.assertIsInstance(token_count, int)
            self.assertTrue(token_count > 0)
            logger.info(f"Doubao token count: {token_count}")
        except InvokeUnsupportedOperationError:
            logger.info("Doubao does not support token counting")
        except Exception as e:
            logger.error(f"Unexpected error during count_tokens for Doubao: {str(e)}")
            self.fail(f"Unexpected error during count_tokens for Doubao: {str(e)}")

    def test_generate(self):
        logger.info("\nTesting generate for Doubao:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        try:
            response = self.sdk.generate(self.default_model, messages)
            self.assertIsInstance(response, Dict)
            self.assertIn('choices', response)
            self.assertIn('message', response['choices'][0])
            logger.info(f"Doubao response: {response['choices'][0]['message']['content']}")
        except InvokeUnsupportedOperationError:
            logger.info("Doubao does not support text generation")
        except Exception as e:
            logger.error(f"Unexpected error during generate for Doubao: {str(e)}")
            self.fail(f"Unexpected error during generate for Doubao: {str(e)}")

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for Doubao:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        try:
            stream = self.sdk.stream_generate(model=self.default_model, messages=messages)
            full_response = ""
            chunk_count = 0
            start_time = time.time()
            timeout = 30  # 30 seconds timeout
            for chunk in stream:
                if time.time() - start_time > timeout:
                    logger.warning("Timeout reached for Doubao")
                    break
                chunk_count += 1
                self.assertIsInstance(chunk, Dict)
                self.assertIn('choices', chunk)
                self.assertIn('message', chunk['choices'][0])
                content = chunk['choices'][0]['message'].get('content', '')
                if content:
                    full_response += content
                    logger.info(f"Doubao chunk {chunk_count}: {content}")
            logger.info(f"\nDoubao full response: {full_response}")
            logger.info(f"Total chunks received: {chunk_count}")
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        except InvokeUnsupportedOperationError:
            logger.info("Doubao does not support stream generation")
        except Exception as e:
            logger.error(f"Unexpected error during stream_generate for Doubao: {str(e)}")
            self.fail(f"Unexpected error during stream_generate for Doubao: {str(e)}")

    def test_list_models(self):
        logger.info("\nTesting list_models for Doubao:")
        try:
            models = self.sdk.list_models()
            self.assertIsInstance(models, List)
            self.assertTrue(len(models) > 0)
            logger.info(f"Doubao models: {models}")
        except InvokeUnsupportedOperationError:
            logger.info("Doubao does not support listing models")
        except Exception as e:
            logger.error(f"Unexpected error while listing models for Doubao: {str(e)}")
            self.fail(f"Unexpected error while listing models for Doubao: {str(e)}")

    def test_tokenize(self):
        logger.info("\nTesting tokenize for Doubao:")
        texts = ["天空为什么这么蓝", "花儿为什么这么香"]
        try:
            result = self.sdk.tokenize(self.default_model, texts)
            self.assertIsInstance(result, Dict)
            self.assertIn('data', result)
            self.assertEqual(len(result['data']), len(texts))

            for item in result['data']:
                self.assertIn('token_ids', item)
                self.assertIn('total_tokens', item)
                self.assertIn('offset_mapping', item)

            logger.info(f"Doubao tokenization result: {result}")
        except InvokeUnsupportedOperationError:
            logger.info("Doubao does not support tokenization")
        except Exception as e:
            logger.error(f"Unexpected error during tokenize for Doubao: {str(e)}")
            self.fail(f"Unexpected error during tokenize for Doubao: {str(e)}")

    @patch.object(OneSDK, 'generate')
    def test_error_handling(self, mock_generate):
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])

    def test_set_model(self):
        logger.info("\nTesting set_model for Doubao:")
        try:
            sdk_with_model = self.sdk.set_model(self.default_model)
            self.assertEqual(sdk_with_model.current_model, self.default_model)
            logger.info(f"Doubao model set successfully: {self.default_model}")
        except Exception as e:
            logger.error(f"Unexpected error during set_model for Doubao: {str(e)}")
            self.fail(f"Unexpected error during set_model for Doubao: {str(e)}")

if __name__ == "__main__":
    unittest.main(verbosity=2)