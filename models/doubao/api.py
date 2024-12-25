from ..base_api import BaseAPI
from typing import List, Dict, Union, Generator
import requests
import json
import os
from urllib.parse import urljoin
from ...logger import logger
from ...utils.error_handler import (
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)

class API(BaseAPI):
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("DOUBAO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable DOUBAO_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Doubao API initialized")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        kwargs['stream'] = True
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        return self._call_api("embeddings", model=model, input=input, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        logger.info(f"Counting tokens for model: {model}")
        response = self._call_api("tokenization", model=model, text=messages)
        token_count = sum(item['total_tokens'] for item in response['data'])
        logger.info(f"Token count for model {model}: {token_count}")
        return token_count

    def create_context(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Create a context for caching."""
        logger.info(f"Creating context for model: {model}")
        return self._call_api("context/create", model=model, messages=messages, **kwargs)

    def generate_with_context(self, model: str, context_id: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using a context."""
        logger.info(f"Generating response with context for model: {model}")
        return self._call_api("context/chat/completions", model=model, context_id=context_id, messages=messages, **kwargs)

    def _call_api(self, endpoint: str, **kwargs):
        url = urljoin(self.BASE_URL, endpoint)
        method = kwargs.pop('method', 'POST')
        stream = kwargs.pop('stream', False)

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            if method == "GET":
                response = self.session.get(url, params=kwargs)
            else:
                response = self.session.post(url, json=kwargs, stream=stream)

            response.raise_for_status()

            if stream:
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