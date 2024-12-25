from ..base_api import BaseAPI
from typing import List, Dict, Union, Generator
import requests
import json
import os
import base64
from urllib.parse import urljoin
from ...logger import logger
from ...utils.error_handler import handle_anthropic_error

class API(BaseAPI):
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"  # Update this as needed

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable ANTHROPIC_API_KEY")
        self.base_url = credentials.get("api_url", self.DEFAULT_BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': self.api_key,
            'anthropic-version': self.API_VERSION,
            'Content-Type': 'application/json'
        })
        logger.info("Anthropic API initialized")

    def list_models(self) -> List[Dict]:
        """List available models."""
        logger.info("Fetching available models")
        models = self._call_api("/v1/models", method="GET")
        logger.info(f"Available models: {[model['id'] for model in models]}")
        return models

    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        logger.info(f"Fetching information for model: {model_id}")
        model_info = self._call_api(f"/v1/models/{model_id}", method="GET")
        logger.info(f"Model info for {model_id}: {model_info}")
        return model_info

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        return self._call_api("/v1/messages", model, messages, stream=False, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        yield from self._call_api("/v1/messages", model, messages, stream=True, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count tokens in a message."""
        logger.info(f"Counting tokens for model: {model}")
        response = self._call_api("/v1/messages/count_tokens", model, messages, count_tokens=True)
        token_count = response['input_tokens']
        logger.info(f"Token count for model {model}: {token_count}")
        return token_count

    def _call_api(self, endpoint: str, model: str = None, data: Union[List, Dict] = None, method: str = "POST",
                  stream: bool = False, count_tokens: bool = False, **kwargs):
        url = urljoin(self.base_url, endpoint)

        headers = self.session.headers.copy()
        if kwargs.get('anthropic_beta'):
            headers['anthropic-beta'] = ','.join(kwargs['anthropic_beta']) if isinstance(kwargs['anthropic_beta'],
                                                                                         list) else kwargs[
                'anthropic_beta']

        if stream:
            headers['Accept'] = 'text/event-stream'

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers)
            else:
                payload = self._prepare_payload(model, data, stream, count_tokens, **kwargs)
                logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
                response = self.session.post(url, json=payload, headers=headers, stream=stream)

            response.raise_for_status()

            if count_tokens:
                return response.json()
            elif stream:
                logger.debug("Received streaming response")
                return self._handle_stream_response(response)
            else:
                logger.debug("Received non-streaming response")
                return self._handle_response(response.json())
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            raise handle_anthropic_error(e)

    def _prepare_payload(self, model: str, data: Union[List, Dict], stream: bool, count_tokens: bool, **kwargs):
        if isinstance(data, list):  # For messages API
            payload = {
                "model": model,
                "messages": self._process_messages(data),
            }
            if not count_tokens:
                payload["stream"] = stream
        else:  # For completion API
            payload = data

        allowed_params = [
            'max_tokens', 'metadata', 'stop_sequences', 'system',
            'temperature', 'top_k', 'top_p', 'tools', 'tool_choice'
        ]
        payload.update({k: v for k, v in kwargs.items() if k in allowed_params})
        logger.debug(f"Prepared payload: {json.dumps(payload, indent=2)}")
        return payload

    def _process_messages(self, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> List[Dict]:
        """Process messages to include image content if present."""
        processed_messages = []
        for message in messages:
            if isinstance(message['content'], list):
                processed_content = []
                for content in message['content']:
                    if content['type'] == 'image':
                        processed_content.append(self._process_image_content(content))
                    else:
                        processed_content.append(content)
                message['content'] = processed_content
            processed_messages.append(message)
        logger.debug(f"Processed messages: {json.dumps(processed_messages, indent=2)}")
        return processed_messages

    def _process_image_content(self, content: Dict) -> Dict:
        """Process image content to base64 if it's a file path."""
        if content['source']['type'] == 'path':
            with open(content['source']['path'], 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            content['source'] = {
                'type': 'base64',
                'media_type': content['source']['media_type'],
                'data': base64_image
            }
        logger.debug(f"Processed image content: {content['source']['type']}")
        return content

    def _handle_response(self, response_data: Dict) -> Dict:
        result = {
            'id': response_data['id'],
            'model': response_data['model'],
            'created': None,  # Anthropic doesn't provide a timestamp
            'choices': [{
                'index': 0,
                'message': response_data['content'][0],
                'finish_reason': response_data['stop_reason']
            }],
            'usage': response_data['usage']
        }
        logger.debug(f"Handled response: {json.dumps(result, indent=2)}")
        return result

    def _handle_stream_response(self, response) -> Generator:
        logger.debug("Entering _handle_stream_response")
        for line in response.iter_lines():
            if line:
                logger.debug(f"Received line: {line.decode('utf-8')}")
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    logger.debug(f"Parsed data: {json.dumps(data, indent=2)}")
                    yield self._handle_response(data)
        logger.debug("Exiting _handle_stream_response")

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")