from ..base_api import BaseAPI
from typing import List, Dict, Union, Generator
import requests
import json
import os
from urllib.parse import urljoin
from ...utils.logger import logger
from ...utils.error_handler import (
    InvokeError,
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)

class API(BaseAPI):
    BASE_URL = "https://generativelanguage.googleapis.com/"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable GEMINI_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        logger.info("Google Gemini API initialized")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate content using the specified model."""
        logger.info(f"Generating content with model: {model}")
        endpoint = f"{model}:generateContent"
        return self._call_api(endpoint, messages=messages, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate streaming content using the specified model."""
        logger.info(f"Generating streaming content with model: {model}")
        endpoint = f"{model}:streamGenerateContent"
        return self._call_api(endpoint, messages=messages, stream=True, **kwargs)

    @BaseAPI.provider_specific
    def generate_content_with_image(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict]]]]]], **kwargs) -> Dict:
        """Generate content with image input using the specified model."""
        logger.info(f"Generating content with image using model: {model}")
        endpoint = f"{model}:generateContent"
        return self._call_api(endpoint, messages=messages, **kwargs)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        endpoint = f"{model}:embedContent"
        return self._call_api(endpoint, content=input, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        logger.info(f"Counting tokens for model: {model}")
        endpoint = f"{model}:countTokens"
        response = self._call_api(endpoint, contents=messages)
        token_count = response.get('totalTokens', 0)
        logger.info(f"Token count for model {model}: {token_count}")
        return token_count

    def _call_api(self, endpoint: str, **kwargs):
        url = urljoin(self.base_url, f"v1/{endpoint}")
        params = {'key': self.api_key}
        stream = kwargs.pop('stream', False)
        
        payload = kwargs

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            if stream:
                response = self.session.post(url, json=payload, params=params, stream=True)
                response.raise_for_status()
                return self._handle_stream_response(response)
            else:
                response = self.session.post(url, json=payload, params=params)
                response.raise_for_status()
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

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")