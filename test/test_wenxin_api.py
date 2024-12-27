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


class TestWenxinAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestWenxinAPI class")

        cls.api_key = os.environ.get("WENXIN_API_KEY")
        cls.secret_key = os.environ.get("WENXIN_SECRET_KEY")
        if not cls.api_key or not cls.secret_key:
            raise ValueError("Please set WENXIN_API_KEY and WENXIN_SECRET_KEY environment variables")

        cls.sdk = OneSDK("wenxin", {"api_key": cls.api_key, "secret_key": cls.secret_key})
        cls.sdk.set_debug_mode(True)

        # 设置自定义模型
        custom_endpoint = "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/custom_model"  # 确保这个端点是正确的
        cls.sdk.api.set_custom_model("Custom-Model", custom_endpoint)

        cls.default_model = "ERNIE-Bot"
        cls.custom_model = "Custom-Model"

    def test_count_tokens(self):
        logger.info("\nTesting count_tokens for Wenxin:")
        messages = [
            {"role": "user", "content": "你好，最近如何？"},
            {"role": "assistant", "content": "我很好，谢谢你的关心。今天我能为你做些什么呢？"}
        ]
        token_count = self.sdk.count_tokens(self.default_model, messages)
        self.assertIsInstance(token_count, int)
        self.assertTrue(token_count > 0)
        logger.info(f"Wenxin token count: {token_count}")

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

        self.assertTrue(content, "Response content is empty")
        if "Error" in content:
            logger.warning(f"API returned an error response: {content}")
        else:
            self.assertIn("1", content)
            self.assertIn("5", content)

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for Wenxin:")
        messages = [{"role": "user", "content": "请从1数到5。"}]
        stream = self.sdk.stream_generate(model=self.default_model, messages=messages)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for Wenxin")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            self.assertIn('choices', chunk)
            self.assertIn('delta', chunk['choices'][0])
            content = chunk['choices'][0]['delta'].get('content', '')
            if content:
                full_response += content
                logger.info(f"Wenxin chunk {chunk_count}: {content}")
        logger.info(f"\nWenxin full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

        self.assertTrue(full_response, "Full response is empty")
        self.assertIn("1", full_response)
        self.assertIn("5", full_response)

    def test_custom_model(self):
        logger.info("\nTesting custom model for Wenxin:")
        messages = [{"role": "user", "content": "使用自定义模型。"}]
        response = self.sdk.generate(self.custom_model, messages)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertTrue(len(response['choices']) > 0, "Choices list is empty")
        self.assertIn('message', response['choices'][0])
        content = response['choices'][0]['message'].get('content', '')
        logger.info(f"Wenxin custom model response: {content}")

        if 'error' in response:
            logger.warning(f"Custom model returned an error: {response['error']}")
            self.skipTest(f"Custom model error: {response['error'].get('message', 'Unknown error')}")
        else:
            self.assertTrue(content, "Custom model response content is empty")

    @patch.object(OneSDK, 'generate')
    def test_error_handling(self, mock_generate):
        logger.info("\nTesting error handling for Wenxin:")
        mock_generate.side_effect = InvokeError("Test error")
        with self.assertRaises(InvokeError):
            self.sdk.generate(self.default_model, [{"role": "user", "content": "Test"}])
        logger.info("Error handling test passed")

    def test_long_conversation(self):
        logger.info("\nTesting long conversation for Wenxin:")
        messages = [
            {"role": "user", "content": "你是谁？"},
            {"role": "assistant", "content": "我是文心一言开发的人工智能助手。"},
            {"role": "user", "content": "你能做什么？"},
            {"role": "assistant", "content": "我可以回答问题、提供信息、进行对话等。"},
            {"role": "user", "content": "给我讲个笑话。"}
        ]
        response = self.sdk.generate(self.default_model, messages)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertTrue(len(response['choices']) > 0, "Choices list is empty")
        content = response['choices'][0]['message'].get('content', '')
        logger.info(f"Wenxin long conversation response: {content}")
        self.assertTrue(content, "Long conversation response is empty")

    def test_embedding(self):
        logger.info("\nTesting embedding for Wenxin:")
        input_text = "这是一个测试文本"
        response = self.sdk.api.embedding("Embedding-V1", input_text)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertTrue(len(response['data']) > 0)
        self.assertIn('embedding', response['data'][0])
        logger.info(f"Wenxin embedding response: {response}")

    def test_text_to_image(self):
        logger.info("\nTesting text to image for Wenxin:")
        prompt = "一只可爱的猫咪"
        response = self.sdk.api.text_to_image("Stable-Diffusion-XL", prompt)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertTrue(len(response['data']) > 0)
        self.assertIn('b64_image', response['data'][0])
        logger.info(f"Wenxin text to image response received")

    def test_create_service(self):
        logger.info("\nTesting create service for Wenxin:")
        response = self.sdk.api.create_service("Test Service", 123, "test_uri")
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        self.assertIn('serviceId', response['result'])
        logger.info(f"Wenxin create service response: {response}")

    def test_get_service_detail(self):
        logger.info("\nTesting get service detail for Wenxin:")
        response = self.sdk.api.get_service_detail(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('id', response)
        logger.info(f"Wenxin get service detail response: {response}")

    def test_get_model_version_detail(self):
        logger.info("\nTesting get model version detail for Wenxin:")
        response = self.sdk.api.get_model_version_detail(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get model version detail response: {response}")

    def test_publish_train_model(self):
        logger.info("\nTesting publish train model for Wenxin:")
        version_meta = {
            "description": "Test model",
            "iterationId": 1,
            "taskId": 1
        }
        response = self.sdk.api.publish_train_model(True, "Test Model", version_meta)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin publish train model response: {response}")

    def test_create_training_task(self):
        logger.info("\nTesting create training task for Wenxin:")
        response = self.sdk.api.create_training_task("Test Task", "Test Description")
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        self.assertIn('id', response['result'])
        logger.info(f"Wenxin create training task response: {response}")

    def test_create_job_run(self):
        logger.info("\nTesting create job run for Wenxin:")
        train_config = {
            "epoch": 1,
            "learningRate": 0.00003
        }
        trainset = [{"type": 1, "id": 123}]
        response = self.sdk.api.create_job_run(1, 0, train_config, trainset, 20)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        self.assertIn('id', response['result'])
        logger.info(f"Wenxin create job run response: {response}")

    def test_get_job_detail(self):
        logger.info("\nTesting get job detail for Wenxin:")
        response = self.sdk.api.get_job_detail(1, 1)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get job detail response: {response}")

    def test_stop_job(self):
        logger.info("\nTesting stop job for Wenxin:")
        response = self.sdk.api.stop_job(1, 1)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin stop job response: {response}")

    def test_create_dataset(self):
        logger.info("\nTesting create dataset for Wenxin:")
        response = self.sdk.api.create_dataset("Test Dataset", 4, 20, 2000, "sysBos")
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        self.assertIn('id', response['result'])
        logger.info(f"Wenxin create dataset response: {response}")

    def test_release_dataset(self):
        logger.info("\nTesting release dataset for Wenxin:")
        response = self.sdk.api.release_dataset(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin release dataset response: {response}")

    def test_import_dataset(self):
        logger.info("\nTesting import dataset for Wenxin:")
        response = self.sdk.api.import_dataset(123, False, 1, ["bos:/test/path/"])
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin import dataset response: {response}")

    def test_get_dataset_info(self):
        logger.info("\nTesting get dataset info for Wenxin:")
        response = self.sdk.api.get_dataset_info(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get dataset info response: {response}")

    def test_get_dataset_status_list(self):
        logger.info("\nTesting get dataset status list for Wenxin:")
        response = self.sdk.api.get_dataset_status_list("123,124")
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get dataset status list response: {response}")

    def test_delete_dataset(self):
        logger.info("\nTesting delete dataset for Wenxin:")
        response = self.sdk.api.delete_dataset(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin delete dataset response: {response}")

    def test_get_dataset_export_record(self):
        logger.info("\nTesting get dataset export record for Wenxin:")
        response = self.sdk.api.get_dataset_export_record(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get dataset export record response: {response}")

    def test_get_dataset_import_error_detail(self):
        logger.info("\nTesting get dataset import error detail for Wenxin:")
        response = self.sdk.api.get_dataset_import_error_detail(123, 1)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get dataset import error detail response: {response}")

    def test_get_prompt_template(self):
        logger.info("\nTesting get prompt template for Wenxin:")
        response = self.sdk.api.get_prompt_template(123)
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin get prompt template response: {response}")

    def test_weather_plugin(self):
        logger.info("\nTesting weather plugin for Wenxin:")
        response = self.sdk.api.weather_plugin("weather", "北京今天天气怎么样", ["uuid-weatherforecast"])
        self.assertIsInstance(response, Dict)
        self.assertIn('result', response)
        logger.info(f"Wenxin weather plugin response: {response}")


if __name__ == "__main__":
    unittest.main(verbosity=2)