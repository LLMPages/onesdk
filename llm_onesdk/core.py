import importlib
import os
from typing import Optional, Union, Generator, List, Dict, Any, Callable
from contextlib import contextmanager

from .utils.error_handler import InvokeConfigError
from .models.base_api import BaseAPI
from .utils.logger import Logger
from .utils.config import Config


class OneSDK:
    """
    A unified interface for interacting with various Large Language Model (LLM) providers.

    This class provides a consistent API for working with different LLM providers,
    abstracting away the differences in their individual APIs.

    Attributes:
        provider (str): The name of the current LLM provider.
        current_model (str): The currently selected model for API calls.

    Example:
        >>> sdk = OneSDK("openai", credentials={"api_key": "your-api-key"})
        >>> sdk.set_model("gpt-3.5-turbo")
        >>> response = sdk.generate(messages=[{"role": "user", "content": "Hello, world!"}])
    """

    def __init__(self, provider: str, credentials: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OneSDK instance.

        Args:
            provider (str): The name of the LLM provider to use.
            credentials (Optional[Dict[str, Any]]): Authentication credentials for the provider.
            config (Optional[Dict[str, Any]]): Additional configuration options.
        """
        self.provider = provider.lower()
        self.config = Config(config or {})
        self.credentials = credentials or {}
        Logger.set_debug_mode(self.config.get('debug', False))
        self.api = self._initialize_api()
        self.current_model = None
        self._register_provider_specific_methods()

    def _initialize_api(self) -> BaseAPI:
        """Initialize the API for the specified provider."""
        try:
            module = importlib.import_module(f'.models.{self.provider}.api', package=__package__)
            api_class = getattr(module, 'API')
            return api_class(self.credentials)
        except (ImportError, AttributeError) as e:
            raise InvokeConfigError(
                f"Unsupported or incorrectly implemented provider: {self.provider}. Error: {str(e)}")

    def _register_provider_specific_methods(self):
        """Register provider-specific methods as attributes of OneSDK."""
        for method_name in self.api.get_provider_specific_methods():
            setattr(self, method_name, self._create_proxy_method(method_name))

    def _create_proxy_method(self, method_name: str) -> Callable:
        """Create a proxy method that calls the corresponding method on the API instance."""

        def proxy_method(*args, **kwargs):
            return getattr(self.api, method_name)(*args, **kwargs)

        return proxy_method

    def call_provider_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method on the provider's API.

        Args:
            method_name (str): The name of the method to call.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            Any: The result of the method call.

        Raises:
            NotImplementedError: If the method is not implemented for the current provider.
        """
        if hasattr(self.api, method_name):
            return getattr(self.api, method_name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"Method '{method_name}' not implemented for provider: {self.provider}")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models for the current provider.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing model information.
        """
        return self.call_provider_method('list_models')

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id (str): The ID of the model to get information about.

        Returns:
            Dict[str, Any]: A dictionary containing model information.
        """
        return self.call_provider_method('get_model', model_id)

    def set_model(self, model: str) -> 'OneSDK':
        """
        Set the current model for subsequent API calls.

        Args:
            model (str): The ID of the model to set as current.

        Returns:
            OneSDK: The current OneSDK instance for method chaining.
        """
        self.current_model = model
        return self

    def generate(self, model: Optional[str] = None, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the specified model or the current model.

        Args:
            model (Optional[str]): The model to use for generation. If None, uses the current model.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments to pass to the provider's API.

        Returns:
            Dict[str, Any]: The generated response from the model.

        Raises:
            InvokeConfigError: If no model is specified and no current model is set.
            InvokeConfigError: If messages is None.

        Example:
            >>> sdk.generate(messages=[{"role": "user", "content": "Tell me a joke"}])
            {'role': 'assistant', 'content': 'Why don't scientists trust atoms? Because they make up everything!'}
        """
        model_to_use = model or self.current_model
        if not model_to_use:
            raise InvokeConfigError("No model specified. Either provide a model parameter or use set_model() method.")
        if messages is None:
            raise InvokeConfigError("Messages cannot be None")
        return self.call_provider_method('generate', model_to_use, messages, **kwargs)

    def stream_generate(self, model: Optional[str] = None,
                        messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = None, **kwargs) -> Generator:
        """
        Generate a streaming response using the specified model or the current model.

        Args:
            model (Optional[str]): The model to use for generation. If None, uses the current model.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments to pass to the provider's API.

        Yields:
            Generator: A generator that yields response chunks from the model.

        Raises:
            InvokeConfigError: If no model is specified and no current model is set.
            InvokeConfigError: If messages is None.
        """
        model_to_use = model or self.current_model
        if not model_to_use:
            raise InvokeConfigError("No model specified. Either provide a model parameter or use set_model() method.")
        if messages is None:
            raise InvokeConfigError("Messages cannot be None")
        yield from self.call_provider_method('stream_generate', model_to_use, messages, **kwargs)

    async def async_generate(self, model: Optional[str] = None,
                             messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = None, **kwargs) -> Dict[
        str, Any]:
        """
        Asynchronously generate a response using the specified model or the current model.

        Args:
            model (Optional[str]): The model to use for generation. If None, uses the current model.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The conversation history.
            **kwargs: Additional keyword arguments to pass to the provider's API.

        Returns:
            Dict[str, Any]: The generated response from the model.

        Raises:
            InvokeConfigError: If no model is specified and no current model is set.
            InvokeConfigError: If messages is None.
        """
        model_to_use = model or self.current_model
        if not model_to_use:
            raise InvokeConfigError("No model specified. Either provide a model parameter or use set_model() method.")
        if messages is None:
            raise InvokeConfigError("Messages cannot be None")
        return await self.call_provider_method('async_generate', model_to_use, messages, **kwargs)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """
        Count the number of tokens in the input messages for the specified model.

        Args:
            model (str): The ID of the model to use for token counting.
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): The messages to count tokens for.

        Returns:
            int: The number of tokens in the input messages.
        """
        return self.call_provider_method('count_tokens', model, messages)

    def create_completion(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Create a completion using the legacy API (if supported by the provider).

        Args:
            model (str): The ID of the model to use for completion.
            prompt (str): The prompt to generate a completion for.
            **kwargs: Additional keyword arguments to pass to the provider's API.

        Returns:
            Dict[str, Any]: The generated completion.
        """
        return self.call_provider_method('create_completion', model, prompt, **kwargs)

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file and return a reference that can be used in messages.

        Args:
            file_path (str): The path to the file to upload.

        Returns:
            str: A reference to the uploaded file that can be used in messages.
        """
        return self.call_provider_method('upload_file', file_path)

    def set_proxy(self, proxy_url: str) -> None:
        """
        Set a proxy for API calls.

        Args:
            proxy_url (str): The URL of the proxy to use.
        """
        return self.call_provider_method('set_proxy', proxy_url)

    def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics for the current account.

        Returns:
            Dict[str, Any]: Usage statistics for the current account.
        """
        return self.call_provider_method('get_usage')

    @staticmethod
    def list_providers() -> List[str]:
        """
        List all available providers.

        Returns:
            List[str]: A list of available provider names.
        """
        providers_dir = os.path.join(os.path.dirname(__file__), 'models')
        return [
            d for d in os.listdir(providers_dir)
            if os.path.isdir(os.path.join(providers_dir, d))  # Check if it's a directory
               and os.path.exists(os.path.join(providers_dir, d, 'api.py'))  # Check if it contains an api.py file
        ]

    def set_debug_mode(self, debug: bool) -> None:
        """
        Set the debug mode for logging.

        Args:
            debug (bool): Whether to enable debug mode.
        """
        Logger.set_debug_mode(debug)
        self.config.set('debug', debug)

    @contextmanager
    def model_context(self, model: str):
        """
        Context manager for temporarily setting a model.

        Args:
            model (str): The ID of the model to use within the context.

        Yields:
            None
        """
        previous_model = self.current_model
        self.set_model(model)
        try:
            yield
        finally:
            self.current_model = previous_model

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Create embeddings for the given input.

        Args:
            model (str): The ID of the model to use for creating embeddings.
            input (Union[str, List[str]]): The input text(s) to create embeddings for.
            **kwargs: Additional keyword arguments to pass to the provider's API.

        Returns:
            Dict[str, Any]: The created embeddings.
        """
        return self.call_provider_method('create_embedding', model, input, **kwargs)

    def create_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Create an image based on the prompt.

        Args:
            prompt (str): The text prompt to generate an image from.
            **kwargs: Additional keyword arguments to pass to the provider's API.

        Returns:
            Dict[str, Any]: Information about the created image.
        """
        return self.call_provider_method('create_image', prompt, **kwargs)

    def custom_operation(self, operation: str, **kwargs) -> Any:
        """
        Perform a custom operation specific to the current provider.

        Args:
            operation (str): The name of the custom operation to perform.
            **kwargs: Additional keyword arguments specific to the operation.

        Returns:
            Any: The result of the custom operation.
        """
        return self.call_provider_method('custom_operation', operation, **kwargs)

    @property
    def cache(self) -> Dict[Any, Any]:
        """
        Get the internal cache dictionary.

        Returns:
            Dict[Any, Any]: The internal cache dictionary.
        """
        if not hasattr(self, '_cache'):
            self._cache = {}
        return self._cache

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
