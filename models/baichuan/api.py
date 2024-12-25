import os
import requests
import json
from typing import List, Dict, Union, Generator
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
from ..base_api import BaseAPI

class API(BaseAPI):
    BASE_URL = "https://api.baichuan-ai.com/v1/"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("BAICHUAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable BAICHUAN_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Baichuan API initialized")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        endpoint = "chat/completions" if model.startswith("Baichuan2") else "chat"
        return self._call_api(endpoint, method="POST", json={
            "model": model,
            "messages": messages,
            **kwargs
        })

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        endpoint = "chat/completions" if model.startswith("Baichuan2") else "stream/chat"
        kwargs['stream'] = True
        response = self._call_api(endpoint, method="POST", json={
            "model": model,
            "messages": messages,
            **kwargs
        }, stream=True)
        return self._handle_stream_response(response)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        return self._call_api("embeddings", method="POST", json={
            "model": model,
            "input": input,
            **kwargs
        })

    def list_files(self) -> List[Dict]:
        """List all files."""
        logger.info("Listing files")
        return self._call_api("files", method="GET")

    def upload_file(self, file_path: str, purpose: str = "knowledge-base") -> Dict:
        """Upload a file."""
        logger.info(f"Uploading file: {file_path}")
        with open(file_path, 'rb') as file:
            files = {'file': file}
            data = {'purpose': purpose}
            return self._call_api("files", method="POST", files=files, data=data)

    def delete_file(self, file_id: str) -> Dict:
        """Delete a file."""
        logger.info(f"Deleting file: {file_id}")
        return self._call_api(f"files/{file_id}", method="DELETE")

    def get_file_info(self, file_id: str) -> Dict:
        """Get information about a file."""
        logger.info(f"Getting file info: {file_id}")
        return self._call_api(f"files/{file_id}", method="GET")

    def get_file_content(self, file_id: str) -> bytes:
        """Get the content of a file."""
        logger.info(f"Getting file content: {file_id}")
        return self._call_api(f"files/{file_id}/content", method="GET", raw_response=True)

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        url = urljoin(self.BASE_URL, endpoint)
        logger.debug(f"Sending request to {url}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()

            if kwargs.get('raw_response'):
                return response.content
            elif kwargs.get('stream'):
                return response
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            raise self._handle_error(e)

    def _handle_stream_response(self, response) -> Generator:
        logger.debug("Entering _handle_stream_response")
        for line in response.iter_lines():
            if line:
                logger.debug(f"Received line: {line.decode('utf-8')}")
                yield json.loads(line.decode('utf-8'))
        logger.debug("Exiting _handle_stream_response")

    def _handle_error(self, error: requests.RequestException) -> InvokeError:
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