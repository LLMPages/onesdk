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
    BASE_URL = "https://api.moonshot.cn/v1/"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable MOONSHOT_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Kimi API initialized")

    def list_models(self) -> List[Dict]:
        """List available models."""
        logger.info("Fetching available models")
        response = self._call_api("models", method="GET")
        models = response.get('data', [])
        logger.info(f"Available models: {[model['id'] for model in models]}")
        return models

    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        logger.info(f"Fetching information for model: {model_id}")
        model_info = self._call_api(f"models/{model_id}", method="GET")
        logger.info(f"Model info for {model_id}: {model_info}")
        return model_info

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        kwargs['stream'] = True
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        logger.info(f"Counting tokens for model: {model}")
        response = self._call_api("chat/completions", model=model, messages=messages, max_tokens=1)
        token_count = response['usage']['prompt_tokens']
        logger.info(f"Token count for model {model}: {token_count}")
        return token_count

    @BaseAPI.provider_specific
    def create_cache(self, model: str, messages: List[Dict], tools: List[Dict] = None, name: str = None,
                     description: str = None, metadata: Dict[str, str] = None, expired_at: int = None,
                     ttl: int = None) -> Dict:
        """Create a context cache."""
        logger.info("Creating context cache")
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "name": name,
            "description": description,
            "metadata": metadata,
        }
        if expired_at:
            payload["expired_at"] = expired_at
        elif ttl:
            payload["ttl"] = ttl
        
        return self._call_api("caching", method="POST", **payload)

    @BaseAPI.provider_specific
    def list_caches(self, limit: int = 20, order: str = "desc", after: str = None, before: str = None,
                    metadata: Dict[str, str] = None) -> Dict:
        """List context caches."""
        logger.info("Listing context caches")
        params = {
            "limit": limit,
            "order": order,
            "after": after,
            "before": before,
        }
        if metadata:
            for key, value in metadata.items():
                params[f"metadata[{key}]"] = value
        
        return self._call_api("caching", method="GET", **params)

    @BaseAPI.provider_specific
    def delete_cache(self, cache_id: str) -> Dict:
        """Delete a context cache."""
        logger.info(f"Deleting context cache: {cache_id}")
        return self._call_api(f"caching/{cache_id}", method="DELETE")

    @BaseAPI.provider_specific
    def update_cache(self, cache_id: str, metadata: Dict[str, str] = None, expired_at: int = None,
                     ttl: int = None) -> Dict:
        """Update a context cache."""
        logger.info(f"Updating context cache: {cache_id}")
        payload = {}
        if metadata:
            payload["metadata"] = metadata
        if expired_at:
            payload["expired_at"] = expired_at
        elif ttl:
            payload["ttl"] = ttl
        
        return self._call_api(f"caching/{cache_id}", method="PUT", **payload)

    @BaseAPI.provider_specific
    def get_cache(self, cache_id: str) -> Dict:
        """Get information about a specific context cache."""
        logger.info(f"Getting context cache: {cache_id}")
        return self._call_api(f"caching/{cache_id}", method="GET")

    @BaseAPI.provider_specific
    def create_tag(self, tag: str, cache_id: str) -> Dict:
        """Create a tag for a context cache."""
        logger.info(f"Creating tag '{tag}' for cache: {cache_id}")
        return self._call_api("caching/refs/tags", method="POST", tag=tag, cache_id=cache_id)

    @BaseAPI.provider_specific
    def list_tags(self, limit: int = 20, order: str = "desc", after: str = None, before: str = None) -> Dict:
        """List tags."""
        logger.info("Listing tags")
        params = {
            "limit": limit,
            "order": order,
            "after": after,
            "before": before,
        }
        return self._call_api("caching/refs/tags", method="GET", **params)

    @BaseAPI.provider_specific
    def delete_tag(self, tag: str) -> Dict:
        """Delete a tag."""
        logger.info(f"Deleting tag: {tag}")
        return self._call_api(f"caching/refs/tags/{tag}", method="DELETE")

    @BaseAPI.provider_specific
    def get_tag(self, tag: str) -> Dict:
        """Get information about a specific tag."""
        logger.info(f"Getting tag: {tag}")
        return self._call_api(f"caching/refs/tags/{tag}", method="GET")

    @BaseAPI.provider_specific
    def get_tag_content(self, tag: str) -> Dict:
        """Get the context cache content for a specific tag."""
        logger.info(f"Getting content for tag: {tag}")
        return self._call_api(f"caching/refs/tags/{tag}/content", method="GET")

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        url = urljoin(self.BASE_URL, endpoint)
        headers = self.session.headers.copy()

        if kwargs.get('stream'):
            headers['Accept'] = 'text/event-stream'

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            if method == "GET":
                response = self.session.get(url, params=kwargs)
            else:
                response = self.session.post(url, json=kwargs)

            response.raise_for_status()

            if kwargs.get('stream'):
                return self._handle_stream_response(response)
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
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    logger.debug(f"Parsed data: {json.dumps(data, indent=2)}")
                    yield data
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