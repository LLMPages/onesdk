import importlib
import os
from typing import Optional, Union, Generator, List, Dict, Any
from contextlib import contextmanager
import asyncio

from .utils.error_handler import InvokeError, InvokeModelNotFoundError, InvokeConfigError
from .models.base_api import BaseAPI
from .utils.logger import Logger
from .utils.config import Config

class OneSDK:
    def __init__(self, provider: str, credentials: Optional[dict] = None, config: Optional[Dict[str, Any]] = None):
        self.provider = provider.lower()
        self.config = Config(config or {})
        self.credentials = credentials or {}
        Logger.set_debug_mode(self.config.get('debug', False))
        self.api = self._initialize_api()
        self.current_model = None
        self._register_provider_specific_methods()

    def _initialize_api(self) -> BaseAPI:
        try:
            module = importlib.import_module(f'.models.{self.provider}.api', package=__package__)
            api_class = getattr(module, 'API')
            return api_class(self.credentials)
        except (ImportError, AttributeError) as e:
            raise InvokeConfigError(f"Unsupported or incorrectly implemented provider: {self.provider}. Error: {str(e)}")

    def _register_provider_specific_methods(self):
        for method_name in self.api.get_provider_specific_methods():
            setattr(self, method_name, self._create_proxy_method(method_name))

    def _create_proxy_method(self, method_name):
        def proxy_method(*args, **kwargs):
            return getattr(self.api, method_name)(*args, **kwargs)
        return proxy_method

    def call_provider_method(self, method_name: str, *args, **kwargs):
        if hasattr(self.api, method_name):
            return getattr(self.api, method_name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"Method '{method_name}' not implemented for provider: {self.provider}")

    def list_models(self) -> List[Dict]:
        """List available models for the current provider."""
        return self.call_provider_method('list_models')

    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        return self.call_provider_method('get_model', model_id)

    def set_model(self, model: str):
        """Set the current model for subsequent API calls."""
        self.current_model = model
        return self

    def generate(self, model: Optional[str] = None, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = None,
                 **kwargs) -> Dict:
        model_to_use = model or self.current_model
        if not model_to_use:
            raise InvokeConfigError("No model specified. Either provide a model parameter or use set_model() method.")
        if messages is None:
            raise InvokeConfigError("Messages cannot be None")
        return self.call_provider_method('generate', model_to_use, messages, **kwargs)

    def stream_generate(self, model: Optional[str] = None,
                        messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = None, **kwargs) -> Generator:
        model_to_use = model or self.current_model
        if not model_to_use:
            raise InvokeConfigError("No model specified. Either provide a model parameter or use set_model() method.")
        if messages is None:
            raise InvokeConfigError("Messages cannot be None")
        yield from self.call_provider_method('stream_generate', model_to_use, messages, **kwargs)

    async def async_generate(self, model: Optional[str] = None,
                             messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = None, **kwargs) -> Dict:
        model_to_use = model or self.current_model
        if not model_to_use:
            raise InvokeConfigError("No model specified. Either provide a model parameter or use set_model() method.")
        if messages is None:
            raise InvokeConfigError("Messages cannot be None")
        return await self.call_provider_method('async_generate', model_to_use, messages, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count the number of tokens in the input messages for the specified model."""
        return self.call_provider_method('count_tokens', model, messages)

    def create_completion(self, model: str, prompt: str, **kwargs) -> Dict:
        """Create a completion using the legacy API (if supported by the provider)."""
        return self.call_provider_method('create_completion', model, prompt, **kwargs)

    def upload_file(self, file_path: str) -> str:
        """Upload a file and return a reference that can be used in messages."""
        return self.call_provider_method('upload_file', file_path)

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        return self.call_provider_method('set_proxy', proxy_url)

    def get_usage(self) -> Dict:
        """Get usage statistics for the current account."""
        return self.call_provider_method('get_usage')

    @staticmethod
    def list_providers() -> List[str]:
        """List all available providers."""
        providers_dir = os.path.join(os.path.dirname(__file__), 'models')
        return [d for d in os.listdir(providers_dir)
                if os.path.isdir(os.path.join(providers_dir, d))
                and os.path.exists(os.path.join(providers_dir, d, 'api.py'))]

    def set_debug_mode(self, debug: bool):
        Logger.set_debug_mode(debug)
        self.config.set('debug', debug)

    @contextmanager
    def model_context(self, model: str):
        """Context manager for temporarily setting a model."""
        previous_model = self.current_model
        self.set_model(model)
        try:
            yield
        finally:
            self.current_model = previous_model

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        return self.call_provider_method('create_embedding', model, input, **kwargs)

    def create_image(self, prompt: str, **kwargs) -> Dict:
        """Create an image based on the prompt."""
        return self.call_provider_method('create_image', prompt, **kwargs)

    def custom_operation(self, operation: str, **kwargs):
        """Perform a custom operation specific to the current provider."""
        return self.call_provider_method('custom_operation', operation, **kwargs)

    @property
    def cache(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
        return self._cache

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()