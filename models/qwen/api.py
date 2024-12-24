from ..base_api import BaseAPI
from typing import List, Dict, Union, Generator
import requests
import json
import os
from urllib.parse import urljoin

from ...utils.error_handler import (
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)


class API(BaseAPI):
    BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    TEXT_GENERATION_ENDPOINT = "text-generation/generation"
    MULTIMODAL_GENERATION_ENDPOINT = "multimodal-generation/generation"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable DASHSCOPE_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

    def list_models(self) -> List[Dict]:
        """List available models."""
        # Aliyun doesn't provide an API to list models, so we'll return a predefined list
        return [
            {"id": "qwen-turbo", "name": "Qwen-Turbo"},
            {"id": "qwen-plus", "name": "Qwen-Plus"},
            {"id": "qwen-max", "name": "Qwen-Max"},
            {"id": "qwen-max-longcontext", "name": "Qwen-Max-LongContext"},
            {"id": "qwen-vl-plus", "name": "Qwen-VL-Plus"},
            # Add more models as needed
        ]

    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        models = self.list_models()
        for model in models:
            if model['id'] == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        endpoint = self._get_endpoint(model)
        return self._call_api(endpoint, model, messages, stream=False, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        endpoint = self._get_endpoint(model)
        return self._call_api(endpoint, model, messages, stream=True, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        # Aliyun doesn't provide a token counting API, so we'll estimate
        # This is a very rough estimate and should be replaced with a more accurate method
        return sum(len(str(message.get('content', '')).split()) for message in messages)

    def _get_endpoint(self, model: str) -> str:
        if model.startswith('qwen-vl') or model.startswith('qwen-audio'):
            return self.MULTIMODAL_GENERATION_ENDPOINT
        return self.TEXT_GENERATION_ENDPOINT

    def _call_api(self, endpoint: str, model: str, messages: List[Dict], stream: bool = False, **kwargs):
        url = urljoin(self.BASE_URL, endpoint)

        payload = self._prepare_payload(model, messages, stream, **kwargs)

        headers = self.session.headers.copy()
        if stream:
            headers['Accept'] = 'text/event-stream'

        try:
            response = self.session.post(url, json=payload, headers=headers, stream=stream)
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return self._handle_response(response.json())
        except requests.RequestException as e:
            self._handle_error(e)

    def _prepare_payload(self, model: str, messages: List[Dict], stream: bool, **kwargs):
        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message"
            }
        }

        # Add optional parameters
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

        return payload

    def _handle_response(self, response_data: Dict) -> Dict:
        if response_data['status_code'] != 200:
            raise InvokeBadRequestError(f"API request failed: {response_data['message']}")

        return {
            'id': response_data['request_id'],
            'model': response_data['output']['choices'][0]['message'].get('role', 'assistant'),
            'created': None,  # Aliyun doesn't provide a timestamp
            'choices': [{
                'index': 0,
                'message': response_data['output']['choices'][0]['message'],
                'finish_reason': response_data['output']['choices'][0]['finish_reason']
            }],
            'usage': response_data['usage']
        }

    def _handle_stream_response(self, response) -> Generator:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    yield self._handle_response(data)

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