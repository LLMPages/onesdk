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

    def setup_credentials(self):
        # Credentials are already set up in __init__, so we can leave this empty
        pass

    def list_models(self) -> List[Dict]:
        """List available models for Doubao."""
        # Implement if Doubao supports listing models, otherwise:
        raise InvokeUnsupportedOperationError("Listing models is not supported by Doubao API")

    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        # Implement if Doubao supports getting model info, otherwise:
        raise InvokeUnsupportedOperationError("Getting model information is not supported by Doubao API")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
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

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        kwargs['stream'] = True
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        payload = {
            "model": model,
            "input": input,
            "encoding_format": kwargs.get("encoding_format", "float")
        }
        return self._call_api("embeddings", **payload)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in messages."""
        logger.info(f"Counting tokens for model: {model}")

        # 提取消息内容，形成文本列表
        text_list = [msg["content"] for msg in messages if isinstance(msg.get("content"), str)]

        payload = {
            "model": model,
            "text": text_list
        }

        try:
            response = self._call_api("tokenization", **payload)
            if 'data' not in response or not response['data']:
                raise InvokeError("Unexpected response format from tokenization API")

            # 计算所有文本的总 token 数
            token_count = sum(item.get('total_tokens', 0) for item in response['data'])
            logger.info(f"Token count for model {model}: {token_count}")
            return token_count
        except Exception as e:
            logger.error(f"Error in count_tokens: {str(e)}")
            return self._fallback_count_tokens(text_list)

    def _fallback_count_tokens(self, text_list: List[str]) -> int:
        """A simple fallback method to estimate token count."""
        total_chars = sum(len(text) for text in text_list)
        # 假设平均每个token包含4个字符
        estimated_tokens = total_chars // 4
        logger.info(f"Estimated token count (fallback method): {estimated_tokens}")
        return estimated_tokens

    @provider_specific
    def create_context(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Create a context for caching."""
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
    def generate_with_context(self, model: str, context_id: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using a context."""
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
    def visual_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using visual understanding."""
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
        url = urljoin(self.BASE_URL, endpoint)
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
        logger.debug("Entering _handle_stream_response")
        for line in response.iter_lines():
            if line:
                logger.debug(f"Received line: {line.decode('utf-8')}")
                yield json.loads(line.decode('utf-8'))
        logger.debug("Exiting _handle_stream_response")

    def _handle_error(self, error: requests.RequestException, error_message: str) -> InvokeError:
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
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")