from ..base_api import BaseAPI, provider_specific
from typing import List, Dict, Union, Generator
import requests
import json
import os
import base64
from urllib.parse import urljoin
from ...utils.logger import logger
from ...utils.error_handler import InvokeError, InvokeConnectionError, InvokeRateLimitError, InvokeAuthorizationError, \
    InvokeBadRequestError

class API(BaseAPI):
    """
    API class for interacting with the Anthropic API.
    Implements the BaseAPI interface for Anthropic-specific functionality.
    """
    BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the Anthropic API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable ANTHROPIC_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': self.api_key,
            'anthropic-version': self.API_VERSION,
            'Content-Type': 'application/json'
        })
        logger.info("Anthropic API initialized")

    @provider_specific
    def list_models(self) -> List[Dict]:
        """
        List available models from Anthropic.

        Returns:
            List[Dict]: A list of dictionaries containing model information.
        """
        logger.info("Fetching available models")
        models = self._call_api("/v1/models", method="GET")
        logger.info(f"Available models: {[model['id'] for model in models['data']]}")
        return models

    @provider_specific
    def get_model(self, model_id: str) -> Dict:
        """
        Get information about a specific model.

        Args:
            model_id (str): The ID of the model to retrieve information for.

        Returns:
            Dict: A dictionary containing model information.
        """
        logger.info(f"Fetching information for model: {model_id}")
        model_info = self._call_api(f"/v1/models/{model_id}", method="GET")
        logger.info(f"Model info for {model_id}: {model_info}")
        return model_info

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """
        Generate a response using the specified model.

        Args:
            model (str): The ID of the model to use for generation.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        logger.info(f"Generating response with model: {model}")
        max_tokens = kwargs.pop('max_tokens', 1000)  # Default value set to 1000, can be adjusted as needed
        return self._call_api("/v1/messages", model=model, messages=messages, max_tokens=max_tokens, stream=False,
                              **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """
        Generate a streaming response using the specified model.

        Args:
            model (str): The ID of the model to use for generation.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Yields:
            Dict: Chunks of the generated response.
        """
        logger.info(f"Generating streaming response with model: {model}")
        max_tokens = kwargs.pop('max_tokens', 1000)
        response = self._call_api("/v1/messages", model=model, messages=messages, max_tokens=max_tokens, stream=True,
                                  **kwargs)
        for chunk in response:
            if 'content' in chunk:
                for content_item in chunk['content']:
                    if content_item['type'] == 'text':
                        yield {'delta': {'text': content_item['text']}}

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """
        Count tokens in a message.

        Args:
            model (str): The ID of the model to use for token counting.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The messages to count tokens for.

        Returns:
            int: The number of tokens in the messages.
        """
        logger.info(f"Counting tokens for model: {model}")
        response = self._call_api("/v1/messages", model=model, messages=messages, max_tokens=1)
        token_count = response.get('usage', {}).get('input_tokens', 0)
        logger.info(f"Token count for model {model}: {token_count}")
        return token_count

    def _call_api(self, endpoint: str, **kwargs) -> Union[Dict, Generator]:
        """
        Make an API call to the Anthropic API.

        Args:
            endpoint (str): The API endpoint to call.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Union[Dict, Generator]: The API response, either as a dictionary or a generator for streaming responses.

        Raises:
            InvokeError: If there's an error during the API call.
        """
        url = urljoin(self.base_url, endpoint)
        headers = self.session.headers.copy()
        method = kwargs.pop('method', 'POST')
        stream = kwargs.pop('stream', False)

        if stream:
            headers['Accept'] = 'text/event-stream'

        try:
            payload = self._prepare_payload(**kwargs) if method == 'POST' else None
            logger.debug(f"Sending request to {url}")
            logger.debug(f"Headers: {headers}")
            if payload:
                logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            response = self.session.request(method, url, json=payload, headers=headers, stream=stream)

            # Log response status code and headers
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")

            # Attempt to log response body, even if status code is not 200
            try:
                if not stream:
                    response_body = response.json()
                    logger.debug(f"Response body: {json.dumps(response_body, indent=2)}")
                else:
                    logger.debug("Streaming response, body not available")
            except json.JSONDecodeError:
                logger.debug(f"Response body (not JSON): {response.text}")

            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Error response status code: {e.response.status_code}")
                logger.error(f"Error response headers: {e.response.headers}")
                try:
                    error_body = e.response.json()
                    logger.error(f"Error response body: {json.dumps(error_body, indent=2)}")
                except json.JSONDecodeError:
                    logger.error(f"Error response body (not JSON): {e.response.text}")
            raise self._handle_request_error(e)

    def _prepare_payload(self, **kwargs) -> Dict:
        """
        Prepare the payload for an API call.

        Args:
            **kwargs: Keyword arguments to include in the payload.

        Returns:
            Dict: The prepared payload.
        """
        payload = {
            "model": kwargs.pop('model'),
            "messages": self._process_messages(kwargs.pop('messages', [])),
            "max_tokens": kwargs.pop('max_tokens', 1),
        }
        allowed_params = ['temperature', 'top_p', 'stop', 'stream', 'metadata']
        payload.update({k: v for k, v in kwargs.items() if k in allowed_params})
        return payload

    def _process_messages(self, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> List[Dict]:
        """
        Process messages, handling any image content.

        Args:
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The messages to process.

        Returns:
            List[Dict]: The processed messages.
        """
        processed_messages = []
        for message in messages:
            if isinstance(message.get('content'), list):
                processed_content = []
                for content in message['content']:
                    if content.get('type') == 'image':
                        processed_content.append(self._process_image_content(content))
                    else:
                        processed_content.append(content)
                message['content'] = processed_content
            processed_messages.append(message)
        return processed_messages

    def _process_image_content(self, content: Dict) -> Dict:
        """
        Process image content, converting file paths to base64-encoded data.

        Args:
            content (Dict): The image content to process.

        Returns:
            Dict: The processed image content.
        """
        if content.get('source', {}).get('type') == 'path':
            with open(content['source']['path'], 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            content['source'] = {
                'type': 'base64',
                'media_type': content['source'].get('media_type', 'image/jpeg'),
                'data': base64_image
            }
        return content

    def _handle_stream_response(self, response: requests.Response) -> Generator:
        """
        Handle a streaming response from the API.

        Args:
            response (requests.Response): The streaming response object.

        Yields:
            Dict: Parsed data from the stream.
        """
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                logger.debug(f"Received line: {line}")
                try:
                    data = json.loads(line)
                    logger.debug(f"Parsed data: {data}")
                    yield data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse streaming response: {line}")

    def _handle_request_error(self, error: requests.RequestException) -> InvokeError:
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