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
    InvokeUnsupportedOperationError
)

class API(BaseAPI):
    """API class for interacting with the Qwen API (DashScope)."""

    BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    TEXT_GENERATION_ENDPOINT = "text-generation/generation"
    MULTIMODAL_GENERATION_ENDPOINT = "multimodal-generation/generation"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the Qwen API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable DASHSCOPE_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Qwen API initialized")

    def list_models(self) -> List[Dict]:
        """
        List available models for Qwen.

        Raises:
            InvokeUnsupportedOperationError: This operation is not supported by Qwen API.
        """
        raise InvokeUnsupportedOperationError("Listing models is not supported by Qwen API")

    def get_model(self, model_id: str) -> Dict:
        """
        Get information about a specific model.

        Args:
            model_id (str): The ID of the model to retrieve information for.

        Raises:
            InvokeUnsupportedOperationError: This operation is not supported by Qwen API.
        """
        raise InvokeUnsupportedOperationError("Getting model information is not supported by Qwen API")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """
        Generate a response using the specified model.

        Args:
            model (str): The model to use for generation.
            messages (List[Dict]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        endpoint = self._get_endpoint(model)
        logger.info(f"Generating response with model: {model}")
        return self._call_api(endpoint, model, messages, stream=False, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """
        Generate a streaming response using the specified model.

        Args:
            model (str): The model to use for generation.
            messages (List[Dict]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Generator: A generator yielding response chunks.
        """
        endpoint = self._get_endpoint(model)
        logger.info(f"Generating streaming response with model: {model}")
        yield from self._call_api(endpoint, model, messages, stream=True, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """
        Count tokens in a message.

        Args:
            model (str): The model to use for token counting.
            messages (List[Dict]): The messages to count tokens for.

        Returns:
            int: The estimated number of tokens in the messages.
        """
        token_count = sum(len(str(message.get('content', '')).split()) for message in messages)
        logger.info(f"Estimated token count for model {model}: {token_count}")
        return token_count

    def _get_endpoint(self, model: str) -> str:
        """
        Determine the appropriate endpoint based on the model.

        Args:
            model (str): The name of the model.

        Returns:
            str: The endpoint URL for the given model.
        """
        if model.startswith('qwen-vl') or model.startswith('qwen-audio'):
            endpoint = self.MULTIMODAL_GENERATION_ENDPOINT
        else:
            endpoint = self.TEXT_GENERATION_ENDPOINT
        logger.debug(f"Using endpoint for model {model}: {endpoint}")
        return endpoint

    def _call_api(self, endpoint: str, model: str, messages: List[Dict], stream: bool = False, **kwargs):
        """
        Make an API call to the Qwen API.

        Args:
            endpoint (str): The API endpoint to call.
            model (str): The model to use.
            messages (List[Dict]): The conversation history.
            stream (bool): Whether to use streaming mode.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Union[Dict, Generator]: The API response, either as a dictionary or a generator for streaming responses.

        Raises:
            InvokeError: If there's an error during the API call.
        """
        url = urljoin(self.base_url, endpoint)
        payload = self._prepare_payload(model, messages, stream, **kwargs)
        headers = self.session.headers.copy()
        if stream:
            headers['Accept'] = 'text/event-stream'

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            response = self.session.post(url, json=payload, headers=headers, stream=stream)
            response.raise_for_status()

            if stream:
                logger.debug("Received streaming response")
                return self._handle_stream_response(response)
            else:
                logger.debug("Received non-streaming response")
                return self._handle_response(response.json())
        except requests.RequestException as e:
            logger.error(f"Error occurred: {str(e)}")
            raise self._handle_error(e)

    def _prepare_payload(self, model: str, messages: List[Dict], stream: bool, **kwargs):
        """
        Prepare the payload for the API call.

        Args:
            model (str): The model to use.
            messages (List[Dict]): The conversation history.
            stream (bool): Whether to use streaming mode.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The prepared payload.
        """
        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message"
            }
        }

        for param in ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_tokens', 'stop', 'seed',
                      'enable_search']:
            if param in kwargs:
                payload['parameters'][param] = kwargs[param]

        if 'system' in kwargs:
            payload['input']['system'] = kwargs['system']

        if 'tools' in kwargs:
            payload['parameters']['tools'] = kwargs['tools']

        if 'tool_choice' in kwargs:
            payload['parameters']['tool_choice'] = kwargs['tool_choice']

        logger.debug(f"Prepared payload: {json.dumps(payload, indent=2)}")
        return payload

    def _handle_response(self, response_data: Dict) -> Dict:
        """
        Handle the API response.

        Args:
            response_data (Dict): The raw API response.

        Returns:
            Dict: The processed response.
        """
        choices = response_data.get('output', {}).get('choices', [])
        if not choices:
            logger.warning("No choices in response")
            return {}

        choice = choices[0]
        result = {
            'id': response_data.get('request_id'),
            'model': 'qwen',
            'created': None,
            'choices': [{
                'index': 0,
                'message': choice.get('message', {}),
                'finish_reason': choice.get('finish_reason')
            }],
            'usage': response_data.get('usage', {})
        }
        logger.debug(f"Handled response: {json.dumps(result, indent=2)}")
        return result

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
                if line.startswith('data:'):
                    data = json.loads(line[5:])
                    logger.debug(f"Parsed data: {json.dumps(data, indent=2)}")
                    yield self._handle_response(data)
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
            logger.error(f"Connection error: {str(error)}")
            return InvokeConnectionError(str(error))
        elif isinstance(error, requests.Timeout):
            logger.error(f"Timeout error: {str(error)}")
            return InvokeConnectionError(str(error))
        elif isinstance(error, requests.HTTPError):
            if error.response.status_code == 429:
                logger.error(f"Rate limit error: {str(error)}")
                return InvokeRateLimitError(str(error))
            elif error.response.status_code in (401, 403):
                logger.error(f"Authorization error: {str(error)}")
                return InvokeAuthorizationError(str(error))
            elif error.response.status_code >= 500:
                logger.error(f"Server unavailable error: {str(error)}")
                return InvokeServerUnavailableError(str(error))
            else:
                logger.error(f"Bad request error: {str(error)}")
                return InvokeBadRequestError(str(error))
        else:
            logger.error(f"Unknown error: {str(error)}")
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