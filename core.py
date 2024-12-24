import importlib
import os
from typing import Optional, Union, Generator, List, Dict
from .utils.error_handler import InvokeError, InvokeModelNotFoundError
from .models.base_api import BaseAPI


class OneSDK:
    def __init__(self, provider: str, credentials: dict):
        self.provider = provider.lower()
        self.credentials = credentials
        self.api = self._initialize_api()

    def _initialize_api(self) -> BaseAPI:
        try:
            # 动态导入 API 模块
            module = importlib.import_module(f'.models.{self.provider}.api', package=__package__)
            # 获取 API 类（现在类名统一为 API）
            api_class = getattr(module, 'API')
            return api_class(self.credentials)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unsupported or incorrectly implemented provider: {self.provider}. Error: {str(e)}")

    def _call_api_method(self, method_name: str, *args, **kwargs):
        if hasattr(self.api, method_name):
            return getattr(self.api, method_name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"Method '{method_name}' not implemented for provider: {self.provider}")

    def list_models(self) -> List[Dict]:
        """List available models for the current provider."""
        return self._call_api_method('list_models')

    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        return self._call_api_method('get_model', model_id)

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        try:
            return self._call_api_method('generate', model, messages, **kwargs)
        except InvokeError as e:
            if isinstance(e, InvokeModelNotFoundError):
                available_models = self.list_models()
                raise InvokeModelNotFoundError(
                    f"Model '{model}' not found. Available models: {[m['id'] for m in available_models]}")
            raise

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        try:
            yield from self._call_api_method('stream_generate', model, messages, **kwargs)
        except InvokeError as e:
            if isinstance(e, InvokeModelNotFoundError):
                available_models = self.list_models()
                raise InvokeModelNotFoundError(
                    f"Model '{model}' not found. Available models: {[m['id'] for m in available_models]}")
            raise

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """Count the number of tokens in the input messages for the specified model."""
        return self._call_api_method('count_tokens', model, messages)

    def create_completion(self, model: str, prompt: str, **kwargs) -> Dict:
        """Create a completion using the legacy API (if supported by the provider)."""
        return self._call_api_method('create_completion', model, prompt, **kwargs)

    def upload_file(self, file_path: str) -> str:
        """Upload a file and return a reference that can be used in messages."""
        return self._call_api_method('upload_file', file_path)

    def set_proxy(self, proxy_url: str):
        """Set a proxy for API calls."""
        return self._call_api_method('set_proxy', proxy_url)

    def get_usage(self) -> Dict:
        """Get usage statistics for the current account."""
        return self._call_api_method('get_usage')

    @staticmethod
    def list_providers() -> List[str]:
        """List all available providers."""
        providers_dir = os.path.join(os.path.dirname(__file__), 'models')
        return [d for d in os.listdir(providers_dir)
                if os.path.isdir(os.path.join(providers_dir, d))
                and os.path.exists(os.path.join(providers_dir, d, 'api.py'))]
