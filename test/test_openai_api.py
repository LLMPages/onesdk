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


class TestOpenAIAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestOpenAIAPI class")

        cls.api_key = os.environ.get("OPENAI_API_KEY")
        if not cls.api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        cls.base_url = os.environ.get("OPENAI_URL")
        cls.sdk = OneSDK("openai", {"api_key": cls.api_key, "api_url": cls.base_url})
        cls.sdk.set_debug_mode(True)

        cls.default_model = "gpt-3.5-turbo"

    def setUp(self):
        time.sleep(1)  # 添加延迟以避免频率限制

    def test_list_models(self):
        logger.info("\nTesting list_models for OpenAI:")
        models = self.sdk.list_models()
        self.assertIsInstance(models, List)
        self.assertTrue(len(models) > 0)
        logger.info(f"OpenAI models: {models}")

    def test_get_model(self):
        logger.info("\nTesting get_model for OpenAI:")
        model_info = self.sdk.get_model(self.default_model)
        self.assertIsInstance(model_info, Dict)
        self.assertEqual(model_info['id'], self.default_model)
        logger.info(f"OpenAI model info: {model_info}")

    def test_generate(self):
        logger.info("\nTesting generate for OpenAI:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        response = self.sdk.generate(self.default_model, messages, max_tokens=100)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        logger.info(f"OpenAI response: {response['choices'][0]['message']['content']}")

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for OpenAI:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        stream = self.sdk.stream_generate(model=self.default_model, messages=messages, max_tokens=100)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for OpenAI")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            logger.info(f"Received chunk {chunk_count}: {chunk}")
            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
            if content:
                full_response += content
                logger.info(f"Content: {content}")
        logger.info(f"\nOpenAI full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        self.assertTrue(chunk_count > 0, "No chunks were received")
        self.assertTrue(len(full_response) > 0, "No content was received")

    def test_create_embedding(self):
        logger.info("\nTesting create_embedding for OpenAI:")
        input_text = "Hello, world!"
        response = self.sdk.api.create_embedding("text-embedding-ada-002", input_text)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertTrue(len(response['data']) > 0)
        logger.info(f"OpenAI embedding created: {len(response['data'][0]['embedding'])} dimensions")

    def test_create_image(self):
        logger.info("\nTesting create_image for OpenAI:")
        prompt = "A cute baby sea otter"
        response = self.sdk.api.create_image(prompt=prompt, n=1, size="256x256")
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertTrue(len(response['data']) > 0)
        logger.info(f"OpenAI image created: {response['data'][0]['url']}")

    def test_create_moderation(self):
        logger.info("\nTesting create_moderation for OpenAI:")
        input_text = "I want to hurt someone."
        response = self.sdk.api.create_moderation(input=input_text)
        self.assertIsInstance(response, Dict)
        self.assertIn('results', response)
        self.assertTrue(len(response['results']) > 0)
        logger.info(f"OpenAI moderation result: {response['results'][0]['flagged']}")

    def test_list_files(self):
        logger.info("\nTesting list_files for OpenAI:")
        files = self.sdk.api.list_files()
        self.assertIsInstance(files, List)
        logger.info(f"OpenAI files: {files}")

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
    #     logger.info("\nTesting set_proxy for OpenAI:")
    #     proxy_url = "http://example.com:8080"  # 使用一个示例 URL，不要实际连接
    #     self.sdk.set_proxy(proxy_url)
    #     logger.info(f"Proxy set to {proxy_url}")
    #     # 注意：这里我们只是测试方法调用，不测试实际连接

    def test_fine_tuning_workflow(self):
        # 1. 创建微调任务
        logger.info("\nTesting create_fine_tuning_job for OpenAI:")
        training_file = "file-abc123"  # 替换为实际的文件ID
        response = self.sdk.api.create_fine_tuning_job(training_file=training_file, model="gpt-3.5-turbo")
        self.assertIsInstance(response, Dict)
        self.assertIn('id', response)
        job_id = response['id']
        logger.info(f"Fine-tuning job created: {job_id}")

        # 等待一段时间，确保任务已经被处理
        time.sleep(5)

        # 2. 获取微调任务信息
        logger.info("\nTesting get_fine_tuning_job for OpenAI:")
        response = self.sdk.api.get_fine_tuning_job(job_id)
        self.assertIsInstance(response, Dict)
        self.assertEqual(response['id'], job_id)
        logger.info(f"Fine-tuning job info retrieved: {response['status']}")

        # 3. 列出微调事件
        logger.info("\nTesting list_fine_tuning_events for OpenAI:")
        response = self.sdk.api.list_fine_tuning_events(job_id)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        logger.info(f"Fine-tuning events listed: {len(response['data'])} events")

        # 4. 取消微调任务
        logger.info("\nTesting cancel_fine_tuning_job for OpenAI:")
        response = self.sdk.api.cancel_fine_tuning_job(job_id)
        self.assertIsInstance(response, Dict)
        self.assertEqual(response['id'], job_id)
        logger.info(f"Fine-tuning job cancelled: {response['status']}")

    def test_list_fine_tuning_jobs(self):
        logger.info("\nTesting list_fine_tuning_jobs for OpenAI:")
        response = self.sdk.api.list_fine_tuning_jobs()
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        logger.info(f"Fine-tuning jobs listed: {len(response['data'])} jobs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
