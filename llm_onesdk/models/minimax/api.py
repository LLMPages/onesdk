import os
import requests
import json
from typing import List, Dict, Union, Generator, BinaryIO, Optional
from urllib.parse import urljoin
from ...utils.error_handler import (
    InvokeError,
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)
from ...utils.logger import logger
from ..base_api import BaseAPI, provider_specific

class API(BaseAPI):
    """API class for interacting with the MiniMax API."""

    BASE_URL = "https://api.minimax.chat/v1/"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the MiniMax API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("MINIMAX_API_KEY")
        self.group_id = credentials.get("group_id") or os.environ.get("MINIMAX_GROUP_ID")
        if not self.api_key or not self.group_id:
            raise ValueError(
                "API key and Group ID must be provided either in credentials or as environment variables MINIMAX_API_KEY and MINIMAX_GROUP_ID")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("MiniMax API initialized")
        logger.debug(f"Base URL: {self.BASE_URL}")

    def generate(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Generate a response using the specified model.

        Args:
            model (str): The model to use for generation.
            messages (List[Dict[str, str]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        logger.info(f"Generating response with model: {model}")

        system_message = {
            "role": "system",
            "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。"
        }

        # Ensure the first message in the list is a system message
        if messages[0].get('role') != 'system':
            messages = [system_message] + messages

        payload = {
            "model": model,
            "messages": messages,
            "tokens_to_generate": kwargs.get('tokens_to_generate', 2048),
            "temperature": kwargs.get('temperature', 0.01),
            "top_p": kwargs.get('top_p', 0.95),
        }
        return self._call_api("chat/completions", method="POST", json=payload)

    def stream_generate(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Generator:
        """
        Generate a streaming response using the specified model.

        Args:
            model (str): The model to use for generation.
            messages (List[Dict[str, str]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Generator: A generator yielding response chunks.
        """
        logger.info(f"Generating streaming response with model: {model}")

        system_message = {
            "role": "system",
            "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。"
        }

        # Ensure the first message in the list is a system message
        if messages[0].get('role') != 'system':
            messages = [system_message] + messages

        payload = {
            "model": model,
            "messages": messages,
            "tokens_to_generate": kwargs.get('tokens_to_generate', 2048),
            "temperature": kwargs.get('temperature', 0.01),
            "top_p": kwargs.get('top_p', 0.95),
            "stream": True
        }
        response = self._call_api("chat/completions", method="POST", json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield {'delta': {'text': delta['content']}}
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse streaming response: {line}")

    def create_embedding(self, model: str, texts: List[str], type: str) -> Dict:
        """
        Create embeddings for the given input.

        Args:
            model (str): The model to use for creating embeddings.
            texts (List[str]): The texts to create embeddings for.
            type (str): The type of embedding to create.

        Returns:
            Dict: The created embeddings.
        """
        logger.info(f"Creating embedding with model: {model}")
        payload = {
            "model": model,
            "texts": texts,
            "type": type
        }
        return self._call_api("embeddings", method="POST", json=payload)

    @provider_specific
    def text_to_speech(self, model: str, text: str, voice_setting: Dict, audio_setting: Dict, **kwargs) -> Dict:
        """
        Convert text to speech using T2A v2 API.

        Args:
            model (str): The model to use for text-to-speech.
            text (str): The text to convert to speech.
            voice_setting (Dict): Voice settings for the conversion.
            audio_setting (Dict): Audio settings for the conversion.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response from the text-to-speech API.
        """
        logger.info(f"Converting text to speech with model: {model}")
        payload = {
            "model": model,
            "text": text,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            **kwargs
        }
        return self._call_api("t2a_v2", method="POST", json=payload)

    @provider_specific
    def create_video_generation_task(self, model: str, prompt: str, **kwargs) -> Dict:
        """
        Create a video generation task.

        Args:
            model (str): The model to use for video generation.
            prompt (str): The prompt for video generation.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response containing the task ID and status.
        """
        logger.info(f"Creating video generation task with prompt: {prompt}")
        payload = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        try:
            return self._call_api("video_generation", method="POST", json=payload)
        except InvokeRateLimitError:
            logger.warning("Rate limit reached for video generation task")
            return {"task_id": "", "base_resp": {"status_code": 1002, "status_msg": "rate limit"}}

    @provider_specific
    def query_video_generation_task(self, task_id: str) -> Dict:
        """
        Query the status of a video generation task.

        Args:
            task_id (str): The ID of the task to query.

        Returns:
            Dict: The status of the video generation task.
        """
        logger.info(f"Querying video generation task: {task_id}")
        return self._call_api(f"query/video_generation", method="GET", params={"task_id": task_id})

    @provider_specific
    def upload_music(self, file: BinaryIO, purpose: str = 'song') -> Dict:
        """
        Upload a music file for voice or instrumental extraction.

        Args:
            file (BinaryIO): The music file to upload.
            purpose (str): The purpose of the upload (default is 'song').

        Returns:
            Dict: The response from the music upload API.
        """
        logger.info(f"Uploading music file for purpose: {purpose}")

        files = {'file': (file.name, file, 'audio/mpeg')}
        data = {'purpose': purpose}

        try:
            response = self._call_api("music_upload", method="POST", files=files, data=data)
            logger.debug(f"Response: {response}")
            return response
        except InvokeError as e:
            logger.error(f"Music upload failed: {str(e)}")
            raise

    @provider_specific
    def generate_music(self, model: str, lyrics: str, refer_voice: Optional[str] = None,
                       refer_instrumental: Optional[str] = None, **kwargs) -> Dict:
        """
        Generate music based on lyrics and optional reference voice and instrumental.

        Args:
            model (str): The model to use for music generation.
            lyrics (str): The lyrics for the music.
            refer_voice (Optional[str]): Reference voice for the music.
            refer_instrumental (Optional[str]): Reference instrumental for the music.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response from the music generation API.
        """
        logger.info(f"Generating music with model: {model}")
        payload = {
            "model": model,
            "lyrics": lyrics,
            "refer_voice": refer_voice,
            "refer_instrumental": refer_instrumental,
            **kwargs
        }
        return self._call_api("music_generation", method="POST", json=payload)

    def list_files(self, purpose: str = None) -> List[Dict]:
        """
        List files that have been uploaded to MiniMax.

        Args:
            purpose (str, optional): Filter files by purpose.

        Returns:
            List[Dict]: A list of dictionaries containing file information.
        """
        logger.info("Listing files")
        params = {"purpose": purpose} if purpose else {}
        response = self._call_api("files/list", method="GET", params=params)
        return response.get('files', [])

    def upload_file(self, file: BinaryIO, purpose: str) -> Dict:
        """
        Upload a file to MiniMax.

        Args:
            file (BinaryIO): The file to upload.
            purpose (str): The purpose of the file upload.

        Returns:
            Dict: The response from the file upload API.

        Raises:
            InvokeError: If the file upload fails.
        """
        logger.info(f"Uploading file for purpose: {purpose}")
        files = {'file': (file.name, file, 'application/octet-stream')}
        data = {'purpose': purpose}
        headers = {'Content-Type': None}  # Let requests set the correct Content-Type
        response = self._call_api("files/upload", method="POST", files=files, data=data, headers=headers)
        if isinstance(response, dict) and 'error' in response:
            raise InvokeError(f"File upload failed: {response['error']}")
        return response

    def delete_file(self, file_id: str) -> Dict:
        """
        Delete a file from MiniMax.

        Args:
            file_id (str): The ID of the file to delete.

        Returns:
            Dict: The response from the file deletion API.
        """
        logger.info(f"Deleting file: {file_id}")
        return self._call_api(f"files/delete", method="POST", json={"file_id": file_id})

    def get_file_info(self, file_id: str) -> Dict:
        """
        Retrieve information about a specific file.

        Args:
            file_id (str): The ID of the file to retrieve information for.

        Returns:
            Dict: Information about the specified file.
        """
        logger.info(f"Retrieving file info: {file_id}")
        return self._call_api(f"files/retrieve", method="GET", params={"file_id": file_id})

    def get_file_content(self, file_id: str) -> bytes:
        """
        Retrieve the content of a specific file.

        Args:
            file_id (str): The ID of the file to retrieve content for.

        Returns:
            bytes: The content of the specified file.
        """
        logger.info(f"Retrieving file content: {file_id}")
        response = self._call_api(f"files/retrieve_content", method="GET", params={"file_id": file_id}, data={})
        return response

    def create_knowledge_base(self, name: str, embedding_model: str, operator_id: int, file_id: Optional[str] = None, **kwargs) -> Dict:
        """
        Create a new knowledge base.

        Args:
            name (str): The name of the knowledge base.
            embedding_model (str): The embedding model to use.
            operator_id (int): The ID of the operator creating the knowledge base.
            file_id (Optional[str]): The ID of a file to associate with the knowledge base.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: Information about the created knowledge base.
        """
        logger.info(f"Creating knowledge base: {name}")
        payload = {
            "operator_id": operator_id,
            "name": name,
            "embedding_model": embedding_model,
            "file_id": file_id,
            **kwargs
        }
        return self._call_api("embedding/create_knowledge_base", method="POST", json=payload)

    def delete_knowledge_base(self, knowledge_base_id: str, operator_id: int) -> Dict:
        """
        Delete a knowledge base.

        Args:
            knowledge_base_id (str): The ID of the knowledge base to delete.
            operator_id (int): The ID of the operator deleting the knowledge base.

        Returns:
            Dict: The response from the knowledge base deletion API.
        """
        logger.info(f"Deleting knowledge base: {knowledge_base_id}")
        return self._call_api("embedding/delete_knowledge_base", method="POST", json={"knowledge_base_id": knowledge_base_id, "operator_id": operator_id})

    def get_knowledge_base(self, knowledge_base_id: str) -> Dict:
        """
        Get information about a specific knowledge base.

        Args:
            knowledge_base_id (str): The ID of the knowledge base to retrieve.

        Returns:
            Dict: Information about the specified knowledge base.
        """
        logger.info(f"Getting knowledge base: {knowledge_base_id}")
        return self._call_api(f"embedding/query_knowledge_base", method="GET", params={"knowledge_base_id": knowledge_base_id})

    def list_knowledge_bases(self, page: int = 0, page_size: int = 10) -> Dict:
        """
        List all knowledge bases.

        Args:
            page (int): The page number for pagination (default is 0).
            page_size (int): The number of items per page (default is 10).

        Returns:
            Dict: A dictionary containing the list of knowledge bases and pagination information.
        """
        logger.info("Listing knowledge bases")
        return self._call_api("embedding/list_knowledge_base", method="GET", params={"page": page, "page_size": page_size})

    def add_document_to_knowledge_base(self, knowledge_base_id: str, file_id: str, operator_id: int, **kwargs) -> Dict:
        """
        Add a document to a knowledge base.

        Args:
            knowledge_base_id (str): The ID of the knowledge base to add the document to.
            file_id (str): The ID of the file to add as a document.
            operator_id (int): The ID of the operator adding the document.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response from the document addition API.
        """
        logger.info(f"Adding document {file_id} to knowledge base {knowledge_base_id}")
        payload = {
            "knowledge_base_id": knowledge_base_id,
            "file_id": file_id,
            "operator_id": operator_id,
            **kwargs
        }
        return self._call_api("embedding/add_document", method="POST", json=payload)

    def delete_document_from_knowledge_base(self, knowledge_base_id: str, file_id: str, operator_id: int) -> Dict:
        """
        Delete a document from a knowledge base.

        Args:
            knowledge_base_id (str): The ID of the knowledge base to delete the document from.
            file_id (str): The ID of the file to delete as a document.
            operator_id (int): The ID of the operator deleting the document.

        Returns:
            Dict: The response from the document deletion API.
        """
        logger.info(f"Deleting document {file_id} from knowledge base {knowledge_base_id}")
        payload = {
            "knowledge_base_id": knowledge_base_id,
            "file_id": file_id,
            "operator_id": operator_id
        }
        return self._call_api("embedding/delete_document", method="POST", json=payload)

    def chatcompletion_pro(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Use the ChatCompletion Pro API.

        Args:
            model (str): The model to use for chat completion.
            messages (List[Dict[str, str]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response from the ChatCompletion Pro API.
        """
        logger.info(f"Using ChatCompletion Pro with model: {model}")
        payload = {
            "model": model,
            "messages": messages,
            "bot_setting": kwargs.get('bot_setting', [{"bot_name": "MM智能助理", "content": "MM智能助理是一款由MiniMax自研的大型语言模型。"}]),
            "reply_constraints": kwargs.get('reply_constraints', {"sender_type": "BOT", "sender_name": "MM智能助理"}),
            "tokens_to_generate": kwargs.get('tokens_to_generate', 2048),
            "temperature": kwargs.get('temperature', 0.01),
            "top_p": kwargs.get('top_p', 0.95),
            **kwargs
        }
        return self._call_api("text/chatcompletion_pro", method="POST", json=payload)

    def stream_chatcompletion_pro(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Generator:
        """
        Use the ChatCompletion Pro API with streaming.

        Args:
            model (str): The model to use for chat completion.
            messages (List[Dict[str, str]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Generator: A generator yielding response chunks.
        """
        logger.info(f"Using streaming ChatCompletion Pro with model: {model}")

        payload = {
            "model": model,
            "messages": messages,
            "bot_setting": kwargs.get('bot_setting', [
                {
                    "bot_name": "MM智能助理",
                    "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。"
                }
            ]),
            "reply_constraints": kwargs.get('reply_constraints', {"sender_type": "BOT", "sender_name": "MM智能助理"}),
            "stream": True,
            "tokens_to_generate": kwargs.get('tokens_to_generate', 2048),
            "temperature": kwargs.get('temperature', 0.01),
            "top_p": kwargs.get('top_p', 0.95),
        }

        response = self._call_api("text/chatcompletion_pro", method="POST", json=payload, stream=True)

        for line in response.iter_lines():
            if line:
                chunk_str = line.decode("utf-8")
                logger.debug(f"Received chunk: {chunk_str}")
                if chunk_str.startswith("data: "):
                    try:
                        parsed_data = json.loads(chunk_str[6:])
                        if "usage" not in parsed_data:  # Ignore the final usage information
                            delta_content = parsed_data["choices"][0]["messages"]
                            yield {'delta': delta_content}
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse streaming response: {chunk_str}")

    @provider_specific
    def create_long_speech_task(self, model: str, text: str, voice_setting: Dict, audio_setting: Dict, **kwargs) -> Dict:
        """
        Create a long speech generation task using T2A Large v2 API.

        Args:
            model (str): The model to use for speech generation.
            text (str): The text to convert to speech.
            voice_setting (Dict): Voice settings for the conversion.
            audio_setting (Dict): Audio settings for the conversion.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response from the long speech generation API.
        """
        logger.info(f"Creating long speech task with model: {model}")
        payload = {
            "model": model,
            "text": text,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            **kwargs
        }
        return self._call_api("t2a_async_v2", method="POST", json=payload)

    @provider_specific
    def query_long_speech_task(self, task_id: str) -> Dict:
        """
        Query the status of a long speech generation task.

        Args:
            task_id (str): The ID of the task to query.

        Returns:
            Dict: The status of the long speech generation task.
        """
        logger.info(f"Querying long speech task: {task_id}")
        return self._call_api(f"query/t2a_async_query_v2", method="GET", params={"task_id": task_id})

    @provider_specific
    def voice_cloning(self, file_id: int, voice_id: str, **kwargs) -> Dict:
        """
        Clone a voice using Voice Cloning API.

        Args:
            file_id (int): The ID of the file to use for voice cloning.
            voice_id (str): The ID to assign to the cloned voice.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The response from the voice cloning API.
        """
        logger.info(f"Cloning voice with file_id: {file_id}")
        payload = {
            "file_id": file_id,
            "voice_id": voice_id,
            **kwargs
        }
        return self._call_api("voice_clone", method="POST", json=payload)

    @provider_specific
    def text_to_voice(self, gender: str, age: str, voice_desc: List[str], text: str) -> Dict:
        """
        Generate a voice based on text description using Voice Generation API.

        Args:
            gender (str): The gender of the voice to generate.
            age (str): The age of the voice to generate.
            voice_desc (List[str]): Description of the voice characteristics.
            text (str): The text to convert to speech.

        Returns:
            Dict: The response from the voice generation API.
        """
        logger.info("Generating voice based on text description")
        payload = {
            "gender": gender,
            "age": age,
            "voice_desc": voice_desc,
            "text": text
        }
        return self._call_api("text2voice", method="POST", json=payload)

    @provider_specific
    def delete_voice(self, voice_type: str, voice_id: str) -> Dict:
        """
        Delete a voice using Delete_Voice API.

        Args:
            voice_type (str): The type of voice to delete.
            voice_id (str): The ID of the voice to delete.

        Returns:
            Dict: The response from the voice deletion API.
        """
        logger.info(f"Deleting voice with voice_id: {voice_id}")
        payload = {
            "voice_type": voice_type,
            "voice_id": voice_id
        }
        return self._call_api("delete_voice", method="POST", json=payload)

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        """
        Make an API call to the MiniMax API.

        Args:
            endpoint (str): The API endpoint to call.
            method (str, optional): The HTTP method to use (default is "POST").
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Union[Dict, Generator]: The API response, either as a dictionary or a generator for streaming responses.

        Raises:
            InvokeError: If there's an error during the API call.
        """
        url = urljoin(self.base_url, endpoint)
        params = kwargs.pop('params', {})
        params['GroupId'] = self.group_id

        headers = self.session.headers.copy()
        if kwargs.get('stream'):
            headers['Accept'] = 'text/event-stream'
        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Params: {params}")
        logger.debug(f"Kwargs: {json.dumps(kwargs, indent=2, default=str)}")

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, **kwargs)
            elif method == "POST":
                if 'files' in kwargs:
                    response = self.session.post(url, headers=headers, params=params, files=kwargs['files'], data=kwargs.get('data'))
                else:
                    response = self.session.post(url, headers=headers, params=params, json=kwargs.get('json'))
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Trace-ID: {response.headers.get('Trace-Id')}")

            response.raise_for_status()

            if kwargs.get('raw_response'):
                return response.content
            elif kwargs.get('stream'):
                return response
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Error response status code: {e.response.status_code}")
                logger.error(f"Error response headers: {e.response.headers}")
                logger.error(f"Error response content: {e.response.text}")
            raise self._handle_error(e)

    def _handle_stream_response(self, response) -> Generator:
        """
        Handle a streaming response from the API.

        Args:
            response (requests.Response): The streaming response object.

        Yields:
            Dict: Parsed JSON data from each line of the stream.
        """
        logger.debug("Entering _handle_stream_response")
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                logger.debug(f"Received line: {line}")
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        logger.debug(f"Parsed data: {json.dumps(data, indent=2)}")
                        yield data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse streaming response: {line}")
        logger.debug("Exiting _handle_stream_response")

    def _handle_error(self, error: requests.RequestException) -> InvokeError:
        """
        Handle errors from API requests.

        Args:
            error (requests.RequestException): The error that occurred during the request.

        Returns:
            InvokeError: An appropriate InvokeError subclass based on the type of error.
        """
        if isinstance(error, requests.ConnectionError):
            return InvokeConnectionError(str(error))
        elif isinstance(error, requests.Timeout):
            return InvokeConnectionError(str(error))
        elif isinstance(error, requests.HTTPError):
            if error.response.status_code == 429:
                return InvokeRateLimitError(str(error))
            elif error.response.status_code in (401, 403):
                return InvokeAuthorizationError(str(error))
            elif error.response.status_code >= 500:
                return InvokeServerUnavailableError(str(error))
            else:
                return InvokeBadRequestError(str(error))
        else:
            return InvokeError(str(error))

    def set_proxy(self, proxy_url: str):
        """
        Set a proxy for API calls.

        Args:
            proxy_url (str): The URL of the proxy to use.
        """
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")