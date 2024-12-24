# models/base_api.py

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Generator

class BaseAPI(ABC):
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials

    @abstractmethod
    def list_models(self) -> List[Dict]:
        """List available models."""
        pass

    @abstractmethod
    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        pass

    @abstractmethod
    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        pass

    @abstractmethod
    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        pass

    @abstractmethod
    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count the number of tokens in the input messages for the specified model."""
        pass

    def create_completion(self, model: str, prompt: str, **kwargs) -> Dict:
        """Create a completion using the legacy API (if supported by the provider)."""
        raise NotImplementedError("Completion API not supported for this provider")

    def upload_file(self, file_path: str) -> str:
        """Upload a file and return a reference that can be used in messages."""
        raise NotImplementedError("File upload not supported for this provider")

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        raise NotImplementedError("Proxy setting not supported for this provider")

    def get_usage(self) -> Dict:
        """Get usage statistics for the current account."""
        raise NotImplementedError("Usage statistics not available for this provider")
