from abc import ABC
from typing import List, Dict, Union, Generator, Any, BinaryIO
import requests
import os
from ..logger import logger
from ..utils.error_handler import (
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeUnsupportedOperationError,
)

class BaseAPI(ABC):
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.session = requests.Session()
        self.setup_credentials()

    def setup_credentials(self):
        """Setup API credentials"""
        raise InvokeUnsupportedOperationError("Credential setup not implemented for this provider")

    def list_models(self) -> List[Dict]:
        """List available models."""
        raise InvokeUnsupportedOperationError("Model listing not supported for this provider")

    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        raise InvokeUnsupportedOperationError("Model info retrieval not supported for this provider")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        raise InvokeUnsupportedOperationError("Text generation not supported for this provider")

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        raise InvokeUnsupportedOperationError("Streaming text generation not supported for this provider")

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count the number of tokens in the input messages for the specified model."""
        raise InvokeUnsupportedOperationError("Token counting not supported for this provider")

    def create_completion(self, model: str, prompt: str, **kwargs) -> Dict:
        """Create a completion using the legacy API (if supported by the provider)."""
        raise InvokeUnsupportedOperationError("Completion API not supported for this provider")

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        raise InvokeUnsupportedOperationError("Embedding API not supported for this provider")

    def create_image(self, prompt: str, **kwargs) -> Dict:
        """Create an image based on the prompt."""
        raise InvokeUnsupportedOperationError("Image creation not supported for this provider")

    def create_edit(self, image: BinaryIO, mask: BinaryIO, prompt: str, **kwargs) -> Dict:
        """Create an edit of an image based on a prompt."""
        raise InvokeUnsupportedOperationError("Image editing not supported for this provider")

    def create_variation(self, image: BinaryIO, **kwargs) -> Dict:
        """Create a variation of an image."""
        raise InvokeUnsupportedOperationError("Image variation not supported for this provider")

    def create_transcription(self, file: BinaryIO, model: str, **kwargs) -> Dict:
        """Transcribe audio to text."""
        raise InvokeUnsupportedOperationError("Audio transcription not supported for this provider")

    def create_translation(self, file: BinaryIO, model: str, **kwargs) -> Dict:
        """Translate audio to English text."""
        raise InvokeUnsupportedOperationError("Audio translation not supported for this provider")

    def create_speech(self, model: str, input: str, voice: str, **kwargs) -> bytes:
        """Generate speech from text."""
        raise InvokeUnsupportedOperationError("Speech generation not supported for this provider")

    def create_moderation(self, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create a moderation for the given input."""
        raise InvokeUnsupportedOperationError("Content moderation not supported for this provider")

    def list_files(self) -> List[Dict]:
        """List files that have been uploaded to the provider."""
        raise InvokeUnsupportedOperationError("File listing not supported for this provider")

    def upload_file(self, file: BinaryIO, purpose: str) -> Dict:
        """Upload a file to the provider."""
        raise InvokeUnsupportedOperationError("File upload not supported for this provider")

    def delete_file(self, file_id: str) -> Dict:
        """Delete a file from the provider."""
        raise InvokeUnsupportedOperationError("File deletion not supported for this provider")

    def get_file_info(self, file_id: str) -> Dict:
        """Retrieve information about a specific file."""
        raise InvokeUnsupportedOperationError("File info retrieval not supported for this provider")

    def get_file_content(self, file_id: str) -> bytes:
        """Retrieve the content of a specific file."""
        raise InvokeUnsupportedOperationError("File content retrieval not supported for this provider")

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        """Base method for making API calls"""
        try:
            response = self.session.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self._handle_error(e)

    def _handle_error(self, error: requests.RequestException):
        """Handle common API errors"""
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

    def _validate_messages(self, messages: List[Dict[str, Any]]):
        """Validate the format of input messages"""
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys")

    @staticmethod
    def _get_env_var(var_name: str, default: str = None) -> str:
        """Safely get an environment variable"""
        return os.environ.get(var_name, default)

    def _log_debug(self, message: str):
        """Log a debug message"""
        logger.debug(message)

    def _log_info(self, message: str):
        """Log an info message"""
        logger.info(message)

    def _log_error(self, message: str):
        """Log an error message"""
        logger.error(message)