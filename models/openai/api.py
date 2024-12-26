import os
import requests
import json
from typing import List, Dict, Union, Generator, BinaryIO
from urllib.parse import urljoin
from ...utils.error_handler import (
    InvokeError,
    InvokeConnectionError,
    InvokeServerUnavailableError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)
from ...utils.logger import logger
from ..base_api import BaseAPI, provider_specific

class API(BaseAPI):
    BASE_URL = "https://api.openai.com/"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either in credentials or as an environment variable OPENAI_API_KEY")
        self.base_url = credentials.get("api_url", self.BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("OpenAI API initialized")

    def list_models(self) -> List[Dict]:
        """List available models."""
        logger.info("Fetching available models")
        response = self._call_api("models", method="GET")
        models = response.get('data', [])
        logger.info(f"Available models: {[model['id'] for model in models]}")
        return models

    @provider_specific
    def get_model(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        logger.info(f"Fetching information for model: {model_id}")
        model_info = self._call_api(f"models/{model_id}", method="GET")
        logger.info(f"Model info for {model_id}: {model_info}")
        return model_info

    def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
        """Generate a response using the specified model."""
        logger.info(f"Generating response with model: {model}")
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                        **kwargs) -> Generator:
        """Generate a streaming response using the specified model."""
        logger.info(f"Generating streaming response with model: {model}")
        kwargs['stream'] = True
        return self._call_api("chat/completions", model=model, messages=messages, **kwargs)

    def create_completion(self, model: str, prompt: str, **kwargs) -> Dict:
        """Create a completion using the legacy API."""
        logger.info(f"Creating completion with model: {model}")
        return self._call_api("completions", model=model, prompt=prompt, **kwargs)

    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create embeddings for the given input."""
        logger.info(f"Creating embedding with model: {model}")
        return self._call_api("embeddings", model=model, input=input, **kwargs)

    @provider_specific
    def create_image(self, prompt: str, **kwargs) -> Dict:
        """Create an image based on the prompt."""
        logger.info(f"Creating image with prompt: {prompt}")
        return self._call_api("images/generations", prompt=prompt, **kwargs)

    @provider_specific
    def create_edit(self, image: BinaryIO, mask: BinaryIO, prompt: str, **kwargs) -> Dict:
        """Create an edit of an image based on a prompt."""
        logger.info(f"Creating image edit with prompt: {prompt}")
        files = {
            'image': ('image.png', image, 'image/png'),
            'mask': ('mask.png', mask, 'image/png')
        }
        data = {'prompt': prompt, **kwargs}
        return self._call_api("images/edits", method="POST", files=files, data=data)

    @provider_specific
    def create_variation(self, image: BinaryIO, **kwargs) -> Dict:
        """Create a variation of an image."""
        logger.info("Creating image variation")
        files = {'image': ('image.png', image, 'image/png')}
        return self._call_api("images/variations", method="POST", files=files, data=kwargs)

    @provider_specific
    def create_transcription(self, file: BinaryIO, model: str, **kwargs) -> Dict:
        """Transcribe audio to text."""
        logger.info(f"Creating transcription with model: {model}")
        files = {'file': ('audio.mp3', file, 'audio/mpeg')}
        data = {'model': model, **kwargs}
        return self._call_api("audio/transcriptions", method="POST", files=files, data=data)

    @provider_specific
    def create_translation(self, file: BinaryIO, model: str, **kwargs) -> Dict:
        """Translate audio to English text."""
        logger.info(f"Creating translation with model: {model}")
        files = {'file': ('audio.mp3', file, 'audio/mpeg')}
        data = {'model': model, **kwargs}
        return self._call_api("audio/translations", method="POST", files=files, data=data)

    @provider_specific
    def create_speech(self, model: str, input: str, voice: str, **kwargs) -> bytes:
        """Generate speech from text."""
        logger.info(f"Creating speech with model: {model}")
        data = {'model': model, 'input': input, 'voice': voice, **kwargs}
        return self._call_api("audio/speech", method="POST", data=data, raw_response=True)

    @provider_specific
    def create_moderation(self, input: Union[str, List[str]], **kwargs) -> Dict:
        """Create a moderation for the given input."""
        logger.info("Creating moderation")
        return self._call_api("moderations", input=input, **kwargs)

    def list_files(self) -> List[Dict]:
        """List files that have been uploaded to OpenAI."""
        logger.info("Listing files")
        response = self._call_api("files", method="GET")
        files = response.get('data', [])
        logger.info(f"Retrieved {len(files)} files")
        return files

    def upload_file(self, file: BinaryIO, purpose: str) -> Dict:
        """Upload a file to OpenAI."""
        logger.info(f"Uploading file for purpose: {purpose}")
        files = {'file': file}
        data = {'purpose': purpose}
        return self._call_api("files", method="POST", files=files, data=data)

    def delete_file(self, file_id: str) -> Dict:
        """Delete a file from OpenAI."""
        logger.info(f"Deleting file: {file_id}")
        return self._call_api(f"files/{file_id}", method="DELETE")

    def get_file_info(self, file_id: str) -> Dict:
        """Retrieve information about a specific file."""
        logger.info(f"Retrieving file info: {file_id}")
        return self._call_api(f"files/{file_id}", method="GET")

    def get_file_content(self, file_id: str) -> bytes:
        """Retrieve the content of a specific file."""
        logger.info(f"Retrieving file content: {file_id}")
        return self._call_api(f"files/{file_id}/content", method="GET", raw_response=True)

    @provider_specific
    def create_fine_tuning_job(self, training_file: str, model: str, **kwargs) -> Dict:
        """Create a fine-tuning job."""
        logger.info(f"Creating fine-tuning job for model: {model}")
        data = {'training_file': training_file, 'model': model, **kwargs}
        return self._call_api("fine_tuning/jobs", method="POST", json=data)

    @provider_specific
    def list_fine_tuning_jobs(self, **kwargs) -> Dict:
        """List fine-tuning jobs."""
        logger.info("Listing fine-tuning jobs")
        return self._call_api("fine_tuning/jobs", method="GET", params=kwargs)

    @provider_specific
    def get_fine_tuning_job(self, job_id: str) -> Dict:
        """Get info about a fine-tuning job."""
        logger.info(f"Getting info for fine-tuning job: {job_id}")
        return self._call_api(f"fine_tuning/jobs/{job_id}", method="GET")

    @provider_specific
    def cancel_fine_tuning_job(self, job_id: str) -> Dict:
        """Cancel a fine-tuning job."""
        logger.info(f"Cancelling fine-tuning job: {job_id}")
        return self._call_api(f"fine_tuning/jobs/{job_id}/cancel", method="POST")

    @provider_specific
    def list_fine_tuning_events(self, job_id: str, **kwargs) -> Dict:
        """List fine-tuning events for a job."""
        logger.info(f"Listing events for fine-tuning job: {job_id}")
        return self._call_api(f"fine_tuning/jobs/{job_id}/events", method="GET", params=kwargs)

    def _call_api(self, endpoint: str, method: str = "POST", **kwargs):
        url = urljoin(self.base_url, 'v1/' + endpoint)
        headers = self.session.headers.copy()

        logger.debug(f"Sending request to {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            if method == "GET":
                response = self.session.get(url, params=kwargs.get('params'))
            elif method == "POST":
                if 'files' in kwargs:
                    response = self.session.post(url, files=kwargs['files'], data=kwargs.get('data'))
                else:
                    response = self.session.post(url, json=kwargs)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if kwargs.get('raw_response'):
                return response.content
            elif kwargs.get('stream'):
                return self._handle_stream_response(response)
            else:
                return response.json()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise self._handle_error(e)

    def _handle_stream_response(self, response) -> Generator:
        logger.debug("Entering _handle_stream_response")
        for line in response.iter_lines():
            if line:
                logger.debug(f"Received line: {line.decode('utf-8')}")
                if line.decode('utf-8').strip() == "data: [DONE]":
                    logger.debug("Received DONE signal, ending stream")
                    break
                try:
                    yield json.loads(line.decode('utf-8').split('data: ')[1])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON: {e}")
                    continue  # Skip this line and continue with the next one
        logger.debug("Exiting _handle_stream_response")

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