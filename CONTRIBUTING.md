# Contributing to OneSDK

We welcome contributions to OneSDK, especially in the form of new provider integrations. This guide will walk you through the process of adding a new provider, such as ChatGPT, to the project.

## Adding a New Provider

To add a new provider to OneSDK, follow these steps:

1. **Create a new directory for the provider**

   In the `models` directory, create a new folder with the name of your provider. For example:

   ```
   onesdk/models/chatgpt/
   ```

2. **Create the API file**

   In the new provider directory, create a file named `api.py`. This file will contain the implementation of the provider's API.

   ```
   onesdk/models/chatgpt/api.py
   ```

3. **Implement the API class**

   In `api.py`, create a class named `API` that inherits from `BaseAPI`. Implement all the required methods from `BaseAPI`. Here's a basic structure:

   ```python
   from ..base_api import BaseAPI
   from typing import List, Dict, Union, Generator

   class API(BaseAPI):
       def __init__(self, credentials: Dict[str, str]):
           super().__init__(credentials)
           # Initialize any provider-specific attributes here

       def list_models(self) -> List[Dict]:
           # Implement method to list available models

       def get_model(self, model_id: str) -> Dict:
           # Implement method to get model information

       def generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Dict:
           # Implement method for text generation

       def stream_generate(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], **kwargs) -> Generator:
           # Implement method for streaming text generation

       def count_tokens(self, model: str, messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> int:
           # Implement method to count tokens

       # Implement other methods as needed
   ```

4. **Handle provider-specific authentication**

   In the `__init__` method, handle the provider's authentication method. This usually involves setting up API keys or other credentials.

5. **Implement error handling**

   Use the custom exception classes from `utils.error_handler` to handle provider-specific errors and map them to OneSDK's standardized error types.

6. **Add tests**

   Create a new test file in the `test` directory, e.g., `test_chatgpt_api.py`, and write unit tests for your new provider implementation.

7. **Update documentation**

   Update the README.md file to include information about the new provider, including any specific setup or usage instructions.

8. **Create a pull request**

   Once you've implemented and tested your new provider, create a pull request with your changes. Include a description of the new provider and any relevant documentation updates.

## Best Practices

- Ensure your implementation adheres to the interface defined in `BaseAPI`.
- Handle rate limiting and other provider-specific constraints appropriately.
- Use type hints to maintain consistency with the rest of the project.
- Write clear, concise comments and docstrings.
- Follow PEP 8 style guidelines for Python code.

## Need Help?

If you have any questions or need assistance while implementing a new provider, please open an issue on the GitHub repository, and we'll be happy to help!

Thank you for contributing to OneSDK!