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
    """
    API class for interacting with the Baichuan AI API.
    Implements the BaseAPI interface for Baichuan-specific functionality.
    """
    BASE_URL = "https://api.baichuan-ai.com/v1/"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the Baichuan API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("BAICHUAN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable BAICHUAN_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Baichuan API initialized")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """
        Generate a response using the specified model.

        Args:
            model (str): The name of the model to use.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        logger.info(f"Generating response with model: {model}")
        endpoint = "chat/completions" if model.startswith("Baichuan2") else "chat"
        return self._call_api(endpoint, method="POST", json={
            "model": model,
            "messages": messages,
            **kwargs
        })

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """
        Generate a streaming response using the specified model.

        Args:
            model (str): The name of the model to use.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Generator: A generator yielding response chunks.
        """
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
        """
        Create embeddings for the given input.

        Args:
            model (str): The name of the model to use for creating embeddings.
            input (Union[str, List[str]]): The text(s) to create embeddings for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The created embeddings.
        """
        logger.info(f"Creating embedding with model: {model}")
        return self._call_api("embeddings", method="POST", json={
            "model": model,
            "input": input
        })

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

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        """
        Make an API call to the Baichuan API.

        Args:
            endpoint (str): The API endpoint to call.
            method (str): The HTTP method to use (default is "POST").
            **kwargs: Additional keyword arguments for the request.

        Returns:
            Union[Dict, requests.Response]: The API response, either as a dictionary or a Response object for streaming.

        Raises:
            InvokeError: If there's an error during the API call.
        """
        url = urljoin(self.base_url, endpoint)
        logger.debug(f"Sending request to {url}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Headers: {self.session.headers}")
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
            # Attempt to output response body
            try:
                error_content = e.response.text if e.response else "No response content"
                logger.error(f"Error response content: {error_content}")
            except AttributeError:
                logger.error("Unable to retrieve error response content")
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
                logger.debug(f"Received line: {line.decode('utf-8')}")
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    line = line[6:]  # Remove "data: " prefix
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {line}")
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