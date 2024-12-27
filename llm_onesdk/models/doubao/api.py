from ..base_api import BaseAPI, provider_specific
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
    """API class for interacting with the Doubao API."""

    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the Doubao API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("DOUBAO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable DOUBAO_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Doubao API initialized")

    def list_models(self) -> List[Dict]:
        """
        List available models for Doubao.

        Raises:
            InvokeUnsupportedOperationError: This operation is not supported by Doubao API.
        """
        raise InvokeUnsupportedOperationError("Listing models is not supported by Doubao API")

    def get_model(self, model_id: str) -> Dict:
        """
        Get information about a specific model.

        Args:
            model_id (str): The ID of the model to retrieve information for.

        Raises:
            InvokeUnsupportedOperationError: This operation is not supported by Doubao API.
        """
        raise InvokeUnsupportedOperationError("Getting model information is not supported by Doubao API")

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
        logger.info(f"Generating response with model: {model}")
        payload = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        optional_params = ["stream_options", "stop", "frequency_penalty", "presence_penalty",
                           "temperature", "top_p", "logprobs", "top_logprobs", "logit_bias"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        return self._call_api("chat/completions", **payload)

    @provider_specific
    def tokenize(self, model: str, text: Union[str, List[str]]) -> Dict:
        """
        Tokenize the given text using the specified model.

        Args:
            model (str): The model to use for tokenization.
            text (Union[str, List[str]]): The text or list of texts to tokenize.

        Returns:
            Dict: A dictionary containing the tokenization results.
        """
        logger.info(f"Tokenizing text with model: {model}")

        if isinstance(text, str):
            text = [text]

        payload = {
            "model": model,
            "text": text
        }

        try:
            response = self._call_api("tokenization", **payload)
            if 'data' not in response or not response['data']:
                raise InvokeError("Unexpected response format from tokenization API")

            logger.info(f"Tokenization completed for {len(text)} text(s)")
            return response
        except Exception as e:
            logger.error(f"Error in tokenize: {str(e)}")
            raise

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
        logger.info(f"Generating streaming response with model: {model}")
        kwargs['stream'] = True
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """
        Create embeddings for the given input.

        Args:
            model (str): The model to use for creating embeddings.
            input (Union[str, List[str]]): The text(s) to create embeddings for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The created embeddings.
        """
        logger.info(f"Creating embedding with model: {model}")
        payload = {
            "model": model,
            "input": input,
            "encoding_format": kwargs.get("encoding_format", "float")
        }
        return self._call_api("embeddings", **payload)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """
        Count tokens in messages.

        Args:
            model (str): The model to use for token counting.
            messages (List[Dict]): The messages to count tokens for.

        Returns:
            int: The number of tokens in the messages.
        """
        logger.info(f"Counting tokens for model: {model}")

        # Extract message content to form a list of texts
        text_list = [msg["content"] for msg in messages if isinstance(msg.get("content"), str)]

        payload = {
            "model": model,
            "text": text_list
        }

        try:
            response = self._call_api("tokenization", **payload)
            if 'data' not in response or not response['data']:
                raise InvokeError("Unexpected response format from tokenization API")

            # Calculate the total token count for all texts
            token_count = sum(item.get('total_tokens', 0) for item in response['data'])
            logger.info(f"Token count for model {model}: {token_count}")
            return token_count
        except Exception as e:
            logger.error(f"Error in count_tokens: {str(e)}")
            return self._fallback_count_tokens(text_list)

    def _fallback_count_tokens(self, text_list: List[str]) -> int:
        """
        A simple fallback method to estimate token count.

        Args:
            text_list (List[str]): List of texts to estimate token count for.

        Returns:
            int: Estimated token count.
        """
        total_chars = sum(len(text) for text in text_list)
        # Assume an average of 4 characters per token
        estimated_tokens = total_chars // 4
        logger.info(f"Estimated token count (fallback method): {estimated_tokens}")
        return estimated_tokens

    @provider_specific
    def create_context(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """
        Create a context for caching.

        Args:
            model (str): The model to use for context creation.
            messages (List[Dict]): The messages to create context for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The created context.
        """
        logger.info(f"Creating context for model: {model}")
        payload = {
            "model": model,
            "messages": messages,
            "mode": kwargs.get("mode", "session"),
            "ttl": kwargs.get("ttl", 86400),
        }
        if "truncation_strategy" in kwargs:
            payload["truncation_strategy"] = kwargs["truncation_strategy"]
        return self._call_api("context/create", **payload)

    @provider_specific
    def generate_with_context(self, model: str, context_id: str,
                              messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """
        Generate a response using a context.

        Args:
            model (str): The model to use for generation.
            context_id (str): The ID of the context to use.
            messages (List[Dict]): The messages to generate a response for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        logger.info(f"Generating response with context for model: {model}")
        payload = {
            "model": model,
            "context_id": context_id,
            "messages": messages,
        }
        optional_params = ["stream", "stream_options", "max_tokens", "stop", "temperature",
                           "top_p", "logprobs", "top_logprobs", "logit_bias"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        return self._call_api("context/chat/completions", **payload)

    @provider_specific
    def visual_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Dict:
        """
        Generate a response using visual understanding.

        Args:
            model (str): The model to use for visual generation.
            messages (List[Dict]): The messages to generate a response for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        logger.info(f"Generating visual response with model: {model}")
        payload = {
            "model": model,
            "messages": messages,
        }
        optional_params = ["stream", "stream_options", "max_tokens", "stop", "temperature",
                           "top_p", "logprobs", "top_logprobs", "logit_bias"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        return self._call_api("chat/completions", **payload)

    def _call_api(self, endpoint: str, **kwargs):
        """
        Make an API call to the Doubao API.

        Args:
            endpoint (str): The API endpoint to call.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Union[Dict, Generator]: The API response, either as a dictionary or a generator for streaming responses.

        Raises:
            InvokeError: If there's an error during the API call.
        """
        url = urljoin(self.base_url, endpoint)
        method = kwargs.pop('method', 'POST')
        stream = kwargs.pop('stream', False)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=kwargs)
            else:
                response = self.session.post(url, headers=headers, json=kwargs, stream=stream)

            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            error_message = "No error message provided"
            if e.response is not None:
                try:
                    error_json = e.response.json()
                    if 'error' in error_json and 'message' in error_json['error']:
                        error_message = error_json['error']['message']
                except json.JSONDecodeError:
                    error_message = e.response.text

            logger.error(f"Error message: {error_message}")
            logger.error(f"Response status code: {e.response.status_code if e.response else 'N/A'}")
            logger.error(f"Response content: {e.response.text if e.response else 'N/A'}")
            raise self._handle_error(e, error_message)

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
                yield json.loads(line.decode('utf-8'))
        logger.debug("Exiting _handle_stream_response")

    def _handle_error(self, error: requests.RequestException, error_message: str) -> InvokeError:
        """
        Handle errors from API requests.

        Args:
            error (requests.RequestException): The error that occurred during the request.
            error_message (str): The error message extracted from the response.

        Returns:
            InvokeError: An appropriate InvokeError subclass based on the type of error.
        """
        if isinstance(error, requests.ConnectionError):
            return InvokeConnectionError(f"{str(error)}. Error message: {error_message}")
        elif isinstance(error, requests.Timeout):
            return InvokeConnectionError(f"{str(error)}. Error message: {error_message}")
        elif isinstance(error, requests.HTTPError):
            if error.response.status_code == 429:
                return InvokeRateLimitError(f"{str(error)}. Error message: {error_message}")
            elif error.response.status_code in (401, 403):
                return InvokeAuthorizationError(f"{str(error)}. Error message: {error_message}")
            elif error.response.status_code >= 500:
                return InvokeServerUnavailableError(f"{str(error)}. Error message: {error_message}")
            else:
                return InvokeBadRequestError(f"{str(error)}. Error message: {error_message}")
        else:
            return InvokeError(f"{str(error)}. Error message: {error_message}")

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