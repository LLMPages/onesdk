from abc import ABC, abstractmethod
from typing import List, Dict, Union, Generator, Any, BinaryIO
import requests
import os
from ..utils.logger import logger
from ..utils.error_handler import (
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeUnsupportedOperationError,
)

def provider_specific(func):
    func._provider_specific = True
    return func

class BaseAPI(ABC):
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.session = requests.Session()
        # self.setup_credentials()

    @abstractmethod
    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        pass

    @abstractmethod
    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        pass

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")

    def custom_operation(self, operation: str, **kwargs):
        """Perform a custom operation specific to this provider."""
        raise InvokeUnsupportedOperationError("Custom operations are not implemented for this provider")

    @classmethod
    def get_provider_specific_methods(cls):
        return [name for name, method in cls.__dict__.items() if getattr(method, '_provider_specific', False)]

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