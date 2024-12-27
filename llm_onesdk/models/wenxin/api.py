from ..base_api import BaseAPI, provider_specific
from typing import List, Dict, Union, Generator
import requests
import json
import os
from ...utils.logger import logger
from ...utils.error_handler import InvokeError, InvokeConnectionError, InvokeRateLimitError, InvokeAuthorizationError, \
    InvokeBadRequestError

class API(BaseAPI):
    """API class for interacting with the Wenxin (Baidu) API."""

    BASE_URL = "https://aip.baidubce.com"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the Wenxin API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("WENXIN_API_KEY")
        self.secret_key = credentials.get("secret_key") or os.environ.get("WENXIN_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "API key and secret key must be provided either in credentials or as environment variables WENXIN_API_KEY and WENXIN_SECRET_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.access_token = self._get_access_token()
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        # Default model endpoint mappings
        self.model_endpoints = {
            "ERNIE-Bot": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
            "ERNIE-Bot-turbo": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
            "BLOOMZ-7B": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/bloomz_7b1"
        }

        logger.info("Wenxin API initialized")

    def _get_access_token(self) -> str:
        """
        Get the access token for API authentication.

        Returns:
            str: The access token.

        Raises:
            InvokeError: If there's an error getting the access token.
        """
        url = f"{self.base_url}/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        try:
            response = requests.get(url, params=params)
            response_body = response.text
            try:
                response_json = response.json()
            except json.JSONDecodeError:
                response_json = {}

            response.raise_for_status()
            return response_json.get("access_token")
        except requests.RequestException as e:
            error_message = f"Error getting access token: {str(e)}\nResponse body: {response_body}"
            logger.error(error_message)
            logger.error(f"Response JSON: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
            raise self._handle_error(e, additional_info=error_message)

    @provider_specific
    def set_custom_model(self, model_name: str, endpoint: str):
        """
        Set a custom model and its corresponding endpoint.

        Args:
            model_name (str): The name of the custom model.
            endpoint (str): The API endpoint for the custom model.
        """
        self.model_endpoints[model_name] = endpoint
        logger.info(f"Custom model '{model_name}' set with endpoint: {endpoint}")

    def _get_endpoint(self, model: str) -> str:
        """
        Get the API endpoint for a given model.

        Args:
            model (str): The name of the model.

        Returns:
            str: The API endpoint for the model.

        Raises:
            ValueError: If the model is not supported.
        """
        if model in self.model_endpoints:
            return self.model_endpoints[model]
        else:
            raise ValueError(
                f"Unsupported model: {model}. Available models are: {', '.join(self.model_endpoints.keys())}")

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

    def _call_api(self, endpoint: str, model: str, messages: List[Dict], stream: bool = False, **kwargs):
        """
        Make an API call to the Wenxin API.

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
        url = f"{self.base_url}{endpoint}?access_token={self.access_token}"
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
                response_json = response.json()
                logger.debug(f"Raw API response: {json.dumps(response_json, indent=2)}")
                return self._handle_response(response_json)
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
            "messages": messages,
            "stream": stream
        }

        for param in ['temperature', 'top_p', 'penalty_score', 'user_id']:
            if param in kwargs:
                payload[param] = kwargs[param]

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
        result = {
            'id': response_data.get('id'),
            'object': response_data.get('object'),
            'created': response_data.get('created'),
            'model': response_data.get('model'),
            'choices': response_data.get('choices', []),
            'usage': response_data.get('usage', {})
        }

        # If choices is empty, add a choice with an error message
        if not result['choices']:
            result['choices'] = [{
                'index': 0,
                'message': {'content': 'Error: No response generated'},
                'finish_reason': 'error'
            }]

        logger.debug(f"Handled response: {json.dumps(result, indent=2)}")
        return result

    def _handle_stream_chunk(self, chunk_data: Dict) -> Dict:
        """
        Handle a single chunk of a streaming response.

        Args:
            chunk_data (Dict): The chunk data.

        Returns:
            Dict: The processed chunk data.
        """
        result = {
            'id': chunk_data.get('id'),
            'object': chunk_data.get('object'),
            'created': chunk_data.get('created'),
            'model': chunk_data.get('model'),
            'choices': [{
                'index': 0,
                'delta': {'content': chunk_data.get('result', '')},
                'finish_reason': chunk_data.get('finish_reason')
            }],
            'usage': chunk_data.get('usage', {})
        }
        logger.debug(f"Handled stream chunk: {json.dumps(result, indent=2)}")
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
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    logger.debug(f"Parsed data: {json.dumps(data, indent=2)}")
                    yield self._handle_stream_chunk(data)
        logger.debug("Exiting _handle_stream_response")

    def _handle_error(self, error: requests.RequestException, additional_info: str = "") -> InvokeError:
        """
        Handle errors from API requests.

        Args:
            error (requests.RequestException): The error that occurred during the request.
            additional_info (str): Additional information about the error.

        Returns:
            InvokeError: An appropriate InvokeError subclass based on the type of error.
        """
        error_message = f"{str(error)}\n{additional_info}"
        if isinstance(error, requests.ConnectionError):
            logger.error(f"Connection error: {error_message}")
            return InvokeConnectionError(error_message)
        elif isinstance(error, requests.Timeout):
            logger.error(f"Timeout error: {error_message}")
            return InvokeConnectionError(error_message)
        elif isinstance(error, requests.HTTPError):
            if error.response.status_code == 429:
                logger.error(f"Rate limit error: {error_message}")
                return InvokeRateLimitError(error_message)
            elif error.response.status_code in (401, 403):
                logger.error(f"Authorization error: {error_message}")
                return InvokeAuthorizationError(error_message)
            elif error.response.status_code >= 500:
                logger.error(f"Server unavailable error: {error_message}")
                return InvokeServerUnavailableError(error_message)
            else:
                logger.error(f"Bad request error: {error_message}")
                return InvokeBadRequestError(error_message)
        else:
            logger.error(f"Unknown error: {error_message}")
            return InvokeError(error_message)

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