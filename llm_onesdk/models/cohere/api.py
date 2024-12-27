import os
import json
from typing import List, Dict, Union, Generator
import requests
from urllib.parse import urljoin
from ...utils.logger import logger
from ..base_api import BaseAPI
from ...utils.error_handler import (
    InvokeError,
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)

class API(BaseAPI):
    """API class for interacting with the Cohere API."""

    BASE_URL = "https://api.cohere.ai/"

    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the Cohere API client.

        Args:
            credentials (Dict[str, str]): A dictionary containing API credentials.
        """
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable COHERE_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("Cohere API initialized")

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """
        Generate a response using the specified model.

        Args:
            model (str): The name of the model to use.
            messages (List[Dict]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The generated response.
        """
        logger.info(f"Generating response with model: {model}")
        endpoint = "generate"
        payload = self._prepare_generate_payload(model, messages, **kwargs)
        return self._call_api(endpoint, payload=payload)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
        """
        Generate a streaming response using the specified model.

        Args:
            model (str): The name of the model to use.
            messages (List[Dict]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Generator: A generator yielding response chunks.
        """
        logger.info(f"Generating streaming response with model: {model}")
        endpoint = "generate"
        payload = self._prepare_generate_payload(model, messages, stream=True, **kwargs)
        return self._call_api(endpoint, payload=payload, stream=True)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """
        Create embeddings for the given input.

        Args:
            model (str): The name of the model to use for creating embeddings.
            input (Union[str, List[str]]): The text(s) to create embeddings for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The created embeddings.
        """
        logger.info(f"Creating embeddings with model: {model}")
        endpoint = "embed"
        payload = {
            "texts": input if isinstance(input, list) else [input],
            "model": model,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
        """
        Count tokens in a message.

        Args:
            model (str): The name of the model to use for token counting.
            messages (List[Dict]): The messages to count tokens for.

        Returns:
            int: The number of tokens in the messages.
        """
        logger.info(f"Counting tokens for model: {model}")
        prompt = self._convert_messages_to_prompt(messages)
        endpoint = "tokenize"
        payload = {
            "text": prompt,
            "model": model
        }
        response = self._call_api(endpoint, payload=payload)
        return len(response.get('tokens', []))

    @BaseAPI.provider_specific
    def chat(self, message: str, chat_history: List[Dict[str, str]] = None, **kwargs) -> Dict:
        """
        Chat with the model.

        Args:
            message (str): The message to send to the model.
            chat_history (List[Dict[str, str]], optional): The chat history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The chat response.
        """
        logger.info("Initiating chat")
        endpoint = "chat"
        payload = {
            "message": message,
            "chat_history": chat_history or [],
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    @BaseAPI.provider_specific
    def classify(self, inputs: List[str], examples: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Classify the given inputs based on examples.

        Args:
            inputs (List[str]): The texts to classify.
            examples (List[Dict[str, str]]): The examples to use for classification.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The classification results.
        """
        logger.info("Classifying inputs")
        endpoint = "classify"
        payload = {
            "inputs": inputs,
            "examples": examples,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    @BaseAPI.provider_specific
    def tokenize(self, text: str, model: str = None, **kwargs) -> Dict:
        """
        Tokenize the given text.

        Args:
            text (str): The text to tokenize.
            model (str, optional): The model to use for tokenization.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The tokenization results.
        """
        logger.info("Tokenizing text")
        endpoint = "tokenize"
        payload = {
            "text": text,
            "model": model,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    @BaseAPI.provider_specific
    def detokenize(self, tokens: List[int], **kwargs) -> Dict:
        """
        Detokenize the given tokens.

        Args:
            tokens (List[int]): The tokens to detokenize.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The detokenization results.
        """
        logger.info("Detokenizing tokens")
        endpoint = "detokenize"
        payload = {
            "tokens": tokens,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    @BaseAPI.provider_specific
    def detect_language(self, texts: List[str], **kwargs) -> Dict:
        """
        Detect the language of the given texts.

        Args:
            texts (List[str]): The texts to detect the language for.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The language detection results.
        """
        logger.info("Detecting language")
        endpoint = "detect-language"
        payload = {
            "texts": texts,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    @BaseAPI.provider_specific
    def summarize(self, text: str, **kwargs) -> Dict:
        """
        Summarize the given text.

        Args:
            text (str): The text to summarize.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The summarization results.
        """
        logger.info("Summarizing text")
        endpoint = "summarize"
        payload = {
            "text": text,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    @BaseAPI.provider_specific
    def rerank(self, query: str, documents: List[str], **kwargs) -> Dict:
        """
        Rerank the given documents based on the query.

        Args:
            query (str): The query to use for reranking.
            documents (List[str]): The documents to rerank.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The reranking results.
        """
        logger.info("Reranking documents")
        endpoint = "rerank"
        payload = {
            "query": query,
            "documents": documents,
            **kwargs
        }
        return self._call_api(endpoint, payload=payload)

    def _prepare_generate_payload(self, model: str, messages: List[Dict], **kwargs):
        """
        Prepare the payload for the generate API call.

        Args:
            model (str): The name of the model to use.
            messages (List[Dict]): The conversation history.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Dict: The prepared payload.
        """
        prompt = self._convert_messages_to_prompt(messages)
        payload = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        return payload

    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert a list of messages to a single prompt string.

        Args:
            messages (List[Dict]): The messages to convert.

        Returns:
            str: The converted prompt string.
        """
        prompt = ""
        for message in messages:
            if message['role'] == 'system':
                prompt += f"System: {message['content']}\n"
            elif message['role'] == 'user':
                prompt += f"Human: {message['content']}\n"
            elif message['role'] == 'assistant':
                prompt += f"Assistant: {message['content']}\n"
        return prompt.strip()

    def _call_api(self, endpoint: str, payload: Dict, method: str = "POST", stream: bool = False):
        """
        Make an API call to the Cohere API.

        Args:
            endpoint (str): The API endpoint to call.
            payload (Dict): The payload for the API call.
            method (str, optional): The HTTP method to use. Defaults to "POST".
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Union[Dict, Generator]: The API response, either as a dictionary or a generator for streaming responses.

        Raises:
            InvokeError: If there's an error during the API call.
        """
        url = urljoin(self.base_url, "v1/" + endpoint)
        headers = self.session.headers.copy()
        if stream:
            headers['Accept'] = 'text/event-stream'

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {payload}")

        try:
            response = self.session.request(method, url, json=payload, headers=headers, stream=stream)
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            raise self._handle_error(e)

    def _handle_stream_response(self, response) -> Generator:
        """
        Handle a streaming response from the API.

        Args:
            response (requests.Response): The streaming response object.

        Yields:
            Dict: Parsed JSON data from each line of the stream.
        """
        for line in response.iter_lines():
            if line:
                yield json.loads(line.decode('utf-8'))

    def _handle_error(self, error: requests.RequestException) -> InvokeError:
        """
        Handle errors from API requests.

        Args:
            error (requests.RequestException): The error that occurred during the request.

        Returns:
            InvokeError: An appropriate InvokeError subclass based on the type of error.
        """
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
        """
        Set a proxy for API calls.

        Args:
            proxy_url (str): The URL of the proxy to use.
        """
        self.session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Proxy set to {proxy_url}")