from ..base_api import BaseAPI
from typing import List, Dict, Union, Generator, Optional
import requests
import json
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
    BASE_URL = "http://localhost:11434"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.base_url = credentials.get("base_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        logger.info("Ollama API initialized")

    def list_models(self) -> List[Dict]:
        """List available models."""
        logger.info("Fetching available models")
        response = self._call_api("/api/tags", method="GET")
        models = response.get('models', [])
        logger.info(f"Available models: {[model['name'] for model in models]}")
        return models

    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        logger.info(f"Fetching information for model: {model_id}")
        response = self._call_api(f"/api/show", method="POST", json={"name": model_id})
        logger.info(f"Model info for {model_id}: {response}")
        return response

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        return self._call_api("/api/generate", model=model, messages=messages, stream=False, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        yield from self._call_api("/api/generate", model=model, messages=messages, stream=True, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        logger.info(f"Counting tokens for model: {model}")
        response = self._call_api("/api/tokenize", model=model, messages=messages)
        token_count = len(response.get('tokens', []))
        logger.info(f"Token count for model {model}: {token_count}")
        return token_count

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        return self._call_api("/api/embeddings", model=model, input=input, **kwargs)

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        url = f"{self.base_url}{endpoint}"
        stream = kwargs.pop('stream', False)
        
        data = {
            "model": kwargs.get('model'),
            "prompt": self._prepare_prompt(kwargs.get('messages', [])),
            "stream": stream,
            **kwargs
        }

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")

        try:
            if method == "GET":
                response = self.session.get(url, params=kwargs.get('params'))
            else:
                response = self.session.post(url, json=data, stream=stream)

            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            raise self._handle_error(e)

    def _prepare_prompt(self, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> str:
        """Prepare prompt from messages."""
        prompt = ""
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            if isinstance(content, list):
                content = ' '.join([item.get('text', '') for item in content if item.get('type') == 'text'])
            prompt += f"{role.capitalize()}: {content}\n"
        return prompt.strip()

    def _handle_stream_response(self, response):
        for line in response.iter_lines():
            if line:
                yield json.loads(line)

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