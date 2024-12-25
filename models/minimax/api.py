import os
import requests
from typing import List, Dict, Union, Generator, BinaryIO
from urllib.parse import urljoin
from ...utils.error_handler import (
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)
from ...logger import logger
from ..base_api import BaseAPI

class API(BaseAPI):
    BASE_URL = "https://api.minimax.chat/v1/"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("MINIMAX_API_KEY")
        self.group_id = credentials.get("group_id") or os.environ.get("MINIMAX_GROUP_ID")
        if not self.api_key or not self.group_id:
            raise ValueError(
                "API key and Group ID must be provided either in credentials or as environment variables MINIMAX_API_KEY and MINIMAX_GROUP_ID")
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("MiniMax API initialized")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        endpoint = "text/chatcompletion_pro" if model == "abab5.5-chat" else "text/chatcompletion"
        return self._call_api(endpoint, model=model, messages=messages, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        kwargs['stream'] = True
        endpoint = "text/chatcompletion_pro" if model == "abab5.5-chat" else "text/chatcompletion"
        return self._call_api(endpoint, model=model, messages=messages, **kwargs)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        return self._call_api("embedding/create_knowledge_base", model=model, input=input, **kwargs)

    def text_to_speech(self, text: str, voice_id: str, **kwargs) -> Dict:
        """Convert text to speech."""
        logger.info(f"Converting text to speech with voice: {voice_id}")
        return self._call_api("text_to_speech", text=text, voice_id=voice_id, **kwargs)

    def create_image(self, prompt: str, **kwargs) -> Dict:
        """Create an image based on the prompt."""
        logger.info(f"Creating image with prompt: {prompt}")
        return self._call_api("images/generations", prompt=prompt, **kwargs)

    def list_files(self, purpose: str = None) -> List[Dict]:
        """List files that have been uploaded to MiniMax."""
        logger.info("Listing files")
        params = {"purpose": purpose} if purpose else {}
        response = self._call_api("files/list", method="GET", params=params)
        files = response.get('files', [])
        logger.info(f"Retrieved {len(files)} files")
        return files

    def upload_file(self, file: BinaryIO, purpose: str) -> Dict:
        """Upload a file to MiniMax."""
        logger.info(f"Uploading file for purpose: {purpose}")
        files = {'file': file}
        data = {'purpose': purpose}
        return self._call_api("files/upload", method="POST", files=files, data=data)

    def delete_file(self, file_id: str) -> Dict:
        """Delete a file from MiniMax."""
        logger.info(f"Deleting file: {file_id}")
        return self._call_api(f"files/delete", method="POST", json={"file_id": file_id})

    def get_file_info(self, file_id: str) -> Dict:
        """Retrieve information about a specific file."""
        logger.info(f"Retrieving file info: {file_id}")
        return self._call_api(f"files/retrieve", method="GET", params={"file_id": file_id})

    def get_file_content(self, file_id: str) -> bytes:
        """Retrieve the content of a specific file."""
        logger.info(f"Retrieving file content: {file_id}")
        return self._call_api(f"files/retrieve_content", method="GET", params={"file_id": file_id}, raw_response=True)

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        url = urljoin(self.BASE_URL, endpoint)
        params = kwargs.pop('params', {})
        params['GroupId'] = self.group_id

        headers = self.session.headers.copy()
        if kwargs.get('stream'):
            headers['Accept'] = 'text/event-stream'

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Params: {params}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, **kwargs)
            elif method == "POST":
                if 'files' in kwargs:
                    response = self.session.post(url, headers=headers, params=params, files=kwargs['files'], data=kwargs.get('data'))
                else:
                    response = self.session.post(url, headers=headers, params=params, json=kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if kwargs.get('raw_response'):
                return response.content
            elif kwargs.get('stream'):
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            self._handle_error(e)

    def _handle_stream_response(self, response) -> Generator:
        logger.debug("Entering _handle_stream_response")
        for line in response.iter_lines():
            if line:
                logger.debug(f"Received line: {line.decode('utf-8')}")
                yield json.loads(line.decode('utf-8'))
        logger.debug("Exiting _handle_stream_response")

    def _handle_error(self, error: requests.RequestException):
        if isinstance(error, requests.ConnectionError):
            raise InvokeConnectionError(str(error))
        elif isinstance(error, requests.Timeout):
            raise InvokeConnectionError(str(error))
        elif isinstance(error, requests.HTTPError):
            if error.response.status_code == 429:
                raise InvokeRateLimitError(str(error))
            elif error.response.status_code in (401, 403):
                raise InvokeAuthorizationError(str(error))
            elif error.response.status_code >= 500:
                raise InvokeServerUnavailableError(str(error))
            else:
                raise InvokeBadRequestError(str(error))
        else:
            raise InvokeBadRequestError(str(error))

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")