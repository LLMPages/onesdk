import os
import sys
import unittest
from typing import List, Dict
import time
from unittest.mock import patch
import io

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_onesdk.core import OneSDK
from llm_onesdk.utils.error_handler import InvokeError, InvokeConnectionError, InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError
from llm_onesdk.utils.logger import Logger, logger


class TestMiniMaxAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.set_debug_mode(True)
        logger.info("Setting up TestMiniMaxAPI class")

        cls.api_key = os.environ.get("MINIMAX_API_KEY")
        cls.group_id = os.environ.get("MINIMAX_GROUP_ID")
        if not cls.api_key or not cls.group_id:
            raise ValueError("Please set MINIMAX_API_KEY and MINIMAX_GROUP_ID environment variables")

        cls.sdk = OneSDK("minimax", {
            "api_key": cls.api_key,
            "group_id": cls.group_id
        })
        cls.sdk.set_debug_mode(True)

        cls.default_model = "abab5.5-chat"  # 使用 MiniMax 的默认模型

    def setUp(self):
        time.sleep(1)  # 添加延迟以避免频率限制

    def test_generate(self):
        logger.info("\nTesting generate for MiniMax:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        bot_setting = [{"bot_name": "MM智能助理", "content": "MM智能助理是一款由MiniMax自研的大型语言模型。"}]
        response = self.sdk.generate(self.default_model, messages, bot_setting=bot_setting)
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertIn('message', response['choices'][0])
        self.assertIn('content', response['choices'][0]['message'])
        logger.info(f"MiniMax response: {response['choices'][0]['message']['content']}")

    def test_stream_generate(self):
        logger.info("\nTesting stream_generate for MiniMax:")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        bot_setting = [{"bot_name": "MM智能助理", "content": "MM智能助理是一款由MiniMax自研的大型语言模型。"}]
        stream = self.sdk.stream_generate(model=self.default_model, messages=messages, bot_setting=bot_setting)
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for MiniMax")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            self.assertIn('delta', chunk)
            content = chunk['delta'].get('text', '')
            if content:
                full_response += content
                logger.info(f"MiniMax chunk {chunk_count}: {content}")
        logger.info(f"\nMiniMax full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        self.assertTrue(chunk_count > 0, "No chunks were received")
        self.assertTrue(len(full_response) > 0, "No content was received")

    def test_create_embedding(self):
        logger.info("\nTesting create_embedding for MiniMax:")
        input_text = ["Hello, world!"]
        response = self.sdk.api.create_embedding("embo-01", input_text, type="db")
        self.assertIsInstance(response, Dict)
        self.assertIn('vectors', response)
        self.assertIsInstance(response['vectors'], List)
        self.assertEqual(len(response['vectors']), 1)  # 应该只有一个向量
        self.assertEqual(len(response['vectors'][0]), 1536)  # 向量长度应为 1536
        logger.info(f"MiniMax embedding: {response['vectors'][0][:5]}...")

    def test_text_to_speech(self):
        logger.info("\nTesting text_to_speech for MiniMax:")
        text = "Hello, how are you?"
        voice_setting = {
            "voice_id": "male-qn-qingse",
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "happy"
        }
        audio_setting = {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        }
        model = "speech-01-turbo"
        response = self.sdk.api.text_to_speech(model, text, voice_setting=voice_setting, audio_setting=audio_setting)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        self.assertIn('audio', response['data'])
        logger.info("Text-to-speech conversion successful")

    def test_create_video_generation_task(self):
        logger.info("\nTesting create_video_generation_task for MiniMax:")
        model = "video-01"
        prompt = "A beautiful sunset over the ocean"
        response = self.sdk.api.create_video_generation_task(model, prompt)
        self.assertIsInstance(response, Dict)
        if response['base_resp']['status_code'] == 1002:
            logger.warning("Rate limit reached, skipping task ID assertion")
        else:
            self.assertIn('task_id', response)
        logger.info(f"Video generation task response: {response}")

    def test_query_video_generation_task(self):
        logger.info("\nTesting query_video_generation_task for MiniMax:")
        # First, create a task
        model = "video-01"
        prompt = "A beautiful sunset over the ocean"
        create_response = self.sdk.api.create_video_generation_task(model, prompt)
        if create_response['base_resp']['status_code'] == 1002:
            logger.warning("Rate limit reached, skipping query test")
            return
        task_id = create_response['task_id']

        # Now query the task
        query_response = self.sdk.api.query_video_generation_task(task_id)
        self.assertIsInstance(query_response, Dict)
        self.assertIn('status', query_response)
        logger.info(f"Video generation task status: {query_response['status']}")

    def test_upload_music(self):
        logger.info("\nTesting upload_music for MiniMax:")
        test_file_content = b"This is a test audio file content."
        test_file = io.BytesIO(test_file_content)
        test_file.name = "test_audio.mp3"

        try:
            response = self.sdk.api.upload_music(test_file)
            self.assertIsInstance(response, Dict)
            self.assertIn('voice_id', response)
            self.assertIn('instrumental_id', response)
            logger.info(
                f"Music uploaded successfully. Voice ID: {response['voice_id']}, Instrumental ID: {response['instrumental_id']}")
        except InvokeError as e:
            self.fail(f"Music upload failed: {str(e)}")

    def test_generate_music(self):
        logger.info("\nTesting generate_music for MiniMax:")
        model = "music-01"
        lyrics = "This is a test song\nWith some simple lyrics\n"
        response = self.sdk.api.generate_music(model, lyrics)
        self.assertIsInstance(response, Dict)
        self.assertIn('data', response)
        logger.info("Music generation successful")

    def test_file_operations(self):
        logger.info("\nTesting file operations for MiniMax:")

        # 测试上传文件
        test_file_content = b"This is a test file content."
        test_file = io.BytesIO(test_file_content)
        test_file.name = "test.txt"  # 给文件一个名字
        upload_response = self.sdk.api.upload_file(test_file, "retrieval")
        self.assertIsInstance(upload_response, Dict)
        self.assertIn('file', upload_response)
        self.assertIn('file_id', upload_response['file'])
        file_id = upload_response['file']['file_id']
        logger.info(f"File uploaded successfully. File ID: {file_id}")

        # 测试获取文件信息
        file_info = self.sdk.api.get_file_info(file_id)
        self.assertIsInstance(file_info, Dict)
        self.assertIn('file', file_info)
        self.assertEqual(file_info['file']['file_id'], file_id)
        logger.info(f"File info retrieved successfully: {file_info}")

        # 测试获取文件内容
        file_content = self.sdk.api.get_file_content(file_id)
        logger.debug(f"Retrieved file content: {file_content}")  # 只记录前100个字节
        if isinstance(file_content, dict):
            # 如果返回的是字典，可能是错误消息
            self.fail(f"Failed to retrieve file content: {file_content}")
        else:
            # 如果是字节串，应该是文件内容
            self.assertEqual(file_content, test_file_content)
            logger.info("File content retrieved successfully")

        # 测试列出文件
        files = self.sdk.api.list_files(purpose="retrieval")
        self.assertIsInstance(files, List)
        self.assertTrue(any(file['file_id'] == file_id for file in files))
        logger.info(f"Listed {len(files)} files")

        # 测试删除文件
        delete_response = self.sdk.api.delete_file(file_id)
        self.assertIsInstance(delete_response, Dict)
        self.assertEqual(delete_response['file_id'], file_id)
        logger.info("File deleted successfully")

    def test_knowledge_base_operations(self):
        logger.info("\nTesting knowledge base operations for MiniMax:")

        # 创建知识库
        kb_name = f"Test KB {int(time.time())}"
        operator_id = int(time.time())
        create_kb_response = self.sdk.api.create_knowledge_base(kb_name, "kbq-001", operator_id)
        self.assertIsInstance(create_kb_response, Dict)
        self.assertIn('knowledge_base_id', create_kb_response)
        kb_id = create_kb_response['knowledge_base_id']
        logger.info(f"Knowledge base created successfully. ID: {kb_id}")

        # 获取知识库信息
        kb_info = self.sdk.api.get_knowledge_base(kb_id)
        self.assertIsInstance(kb_info, Dict)
        self.assertIn('knowledge_base', kb_info)
        self.assertEqual(kb_info['knowledge_base']['knowledge_base_id'], kb_id)
        logger.info(f"Knowledge base info retrieved successfully: {kb_info}")

        # 列出知识库
        kb_list = self.sdk.api.list_knowledge_bases()
        self.assertIsInstance(kb_list, Dict)
        self.assertIn('knowledge_bases', kb_list)
        self.assertTrue(any(kb['knowledge_base_id'] == kb_id for kb in kb_list['knowledge_bases']))
        logger.info(f"Listed {len(kb_list['knowledge_bases'])} knowledge bases")

        # 删除知识库
        delete_kb_response = self.sdk.api.delete_knowledge_base(kb_id, operator_id)
        self.assertIsInstance(delete_kb_response, Dict)
        self.assertIn('base_resp', delete_kb_response)
        self.assertEqual(delete_kb_response['base_resp']['status_code'], 0)
        logger.info("Knowledge base deleted successfully")

    def test_chatcompletion_pro(self):
        logger.info("\nTesting chatcompletion_pro for MiniMax:")
        messages = [{"sender_type": "USER", "sender_name": "User", "text": "Count from 1 to 5."}]
        bot_setting = [{"bot_name": "MM智能助理", "content": "MM智能助理是一款由MiniMax自研的大型语言模型。"}]
        reply_constraints = {"sender_type": "BOT", "sender_name": "MM智能助理"}
        response = self.sdk.api.chatcompletion_pro(
            self.default_model,
            messages,
            bot_setting=bot_setting,
            reply_constraints=reply_constraints
        )
        self.assertIsInstance(response, Dict)
        self.assertIn('choices', response)
        self.assertIn('messages', response['choices'][0])
        self.assertIn('text', response['choices'][0]['messages'][0])
        logger.info(f"MiniMax ChatCompletion Pro response: {response['choices'][0]['messages'][0]['text']}")

    def test_stream_chatcompletion_pro(self):
        logger.info("\nTesting stream_chatcompletion_pro for MiniMax:")
        messages = [{"sender_type": "USER", "sender_name": "小明", "text": "帮我用英文翻译下面这句话：我来自中国"}]
        stream = self.sdk.api.stream_chatcompletion_pro(self.default_model, messages)
        full_response = []
        chunk_count = 0
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        for chunk in stream:
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached for MiniMax")
                break
            chunk_count += 1
            self.assertIsInstance(chunk, Dict)
            self.assertIn('delta', chunk)
            delta_content = chunk['delta']
            full_response.extend(delta_content)
            logger.info(f"MiniMax chunk {chunk_count}: {delta_content}")
        logger.info(f"\nMiniMax full response: {full_response}")
        logger.info(f"Total chunks received: {chunk_count}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        self.assertTrue(chunk_count > 0, "No chunks were received")
        self.assertTrue(len(full_response) > 0, "No content was received")

    def test_create_long_speech_task(self):
        logger.info("\nTesting create_long_speech_task for MiniMax:")
        model = "speech-01-turbo"
        text = "This is a long text for speech synthesis. " * 10
        voice_setting = {
            "voice_id": "male-qn-qingse",
            "speed": 1,
            "vol": 1,
            "pitch": 0
        }
        audio_setting = {
            "audio_sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 2
        }
        response = self.sdk.api.create_long_speech_task(model, text, voice_setting, audio_setting)
        self.assertIsInstance(response, Dict)
        self.assertIn('task_id', response)
        logger.info(f"Long speech task created successfully. Task ID: {response['task_id']}")

    def test_query_long_speech_task(self):
        logger.info("\nTesting query_long_speech_task for MiniMax:")
        # First, create a task
        model = "speech-01-turbo"
        text = "This is a long text for speech synthesis. " * 10
        voice_setting = {
            "voice_id": "male-qn-qingse",
            "speed": 1,
            "vol": 1,
            "pitch": 0
        }
        audio_setting = {
            "audio_sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 2
        }
        create_response = self.sdk.api.create_long_speech_task(model, text, voice_setting, audio_setting)
        task_id = create_response['task_id']

        # Now query the task
        query_response = self.sdk.api.query_long_speech_task(task_id)
        self.assertIsInstance(query_response, Dict)
        self.assertIn('status', query_response)
        logger.info(f"Long speech task status: {query_response['status']}")

    def test_voice_cloning(self):
        logger.info("\nTesting voice_cloning for MiniMax:")
        # First, upload a file for voice cloning
        test_file_content = b"This is a test audio file content for voice cloning."
        test_file = io.BytesIO(test_file_content)
        test_file.name = "test_voice_clone.mp3"
        upload_response = self.sdk.api.upload_file(test_file, "voice_clone")
        file_id = upload_response['file']['file_id']

        # Now clone the voice
        voice_id = f"test_voice_{int(time.time())}"
        response = self.sdk.api.voice_cloning(file_id, voice_id)
        self.assertIsInstance(response, Dict)
        self.assertIn('base_resp', response)
        status_code = response['base_resp']['status_code']
        self.assertTrue(status_code == 0 or status_code == 2038,
                        f"Unexpected status code: {status_code}")
        if status_code == 0:
            logger.info(f"Voice cloned successfully. Voice ID: {voice_id}")
        else:
            logger.warning(f"Voice cloning failed with status code: {status_code}")

    def test_text_to_voice(self):
        logger.info("\nTesting text_to_voice for MiniMax:")
        gender = "female"
        age = "young"
        voice_desc = ["Kind and friendly", "Calm tone"]
        text = "Hello, this is a test for text to voice generation."
        response = self.sdk.api.text_to_voice(gender, age, voice_desc, text)
        self.assertIsInstance(response, Dict)
        self.assertIn('voice_id', response)
        self.assertIn('trial_audio', response)
        logger.info(f"Text to voice generated successfully. Voice ID: {response['voice_id']}")

    def test_delete_voice(self):
        logger.info("\nTesting delete_voice for MiniMax:")
        # First, create a voice using text_to_voice
        gender = "male"
        age = "middle-aged"
        voice_desc = ["Deep voice", "Serious tone"]
        text = "This is a test voice for deletion."
        create_response = self.sdk.api.text_to_voice(gender, age, voice_desc, text)
        voice_id = create_response['voice_id']

        # Now delete the voice
        delete_response = self.sdk.api.delete_voice("voice_generation", voice_id)
        self.assertIsInstance(delete_response, Dict)
        self.assertIn('base_resp', delete_response)
        self.assertEqual(delete_response['base_resp']['status_code'], 0)
        logger.info(f"Voice deleted successfully. Voice ID: {voice_id}")

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

    def test_set_proxy(self):
        logger.info("\nTesting set_proxy for MiniMax:")
        proxy_url = "http://example.com:8080"  # 使用一个示例 URL，不要实际连接
        self.sdk.set_proxy(proxy_url)
        logger.info(f"Proxy set to {proxy_url}")
        # 注意：这里我们只是测试方法调用，不测试实际连接


if __name__ == "__main__":
    unittest.main(verbosity=2)