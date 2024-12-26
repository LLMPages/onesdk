# Baichuan API Documentation

## Overview

Baichuan provides access to its advanced language models through an API. This document outlines how to use the Baichuan API within the OneSDK framework, offering a seamless integration for various natural language processing tasks.

## Basic Usage

### Initialization

To use the Baichuan API, initialize the OneSDK with your Baichuan API key:

```python
from llm_onesdk import OneSDK

baichuan_sdk = OneSDK("baichuan", {
    "api_key": "your_api_key_here",
    "api_url": "https://api.baichuan-ai.com/v1/"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key as an environment variable `BAICHUAN_API_KEY`, and the SDK will automatically use it.

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "Baichuan2-53B"  # Or another available Baichuan model
messages = [{"role": "user", "content": "What are the main features of quantum computing?"}]

response = baichuan_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

Note: The endpoint used depends on the model. For models starting with "Baichuan2", it uses "chat/completions", otherwise "chat".

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in baichuan_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

Note: The streaming endpoint is "chat/completions" for Baichuan2 models and "stream/chat" for others.

### Creating Embeddings

To create embeddings for text input:

```python
model = "Baichuan-embedding-model"  # Replace with actual embedding model name
input_text = "Your text here"
embedding_response = baichuan_sdk.create_embedding(model, input_text)
print(embedding_response['data'][0]['embedding'])
```

## Advanced Features

### Setting a Proxy

To use a proxy for API calls:

```python
baichuan_sdk.set_proxy("http://your-proxy-url:port")
```

### Changing the API Base URL

If you need to use a different API endpoint, you can specify a custom base URL during initialization:

```python
baichuan_sdk = OneSDK("baichuan", {
    "api_key": "your_api_key_here",
    "api_url": "https://your-custom-endpoint.com/v1/"
})
```

## Error Handling

The SDK raises `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import InvokeError, InvokeConnectionError, InvokeServerUnavailableError, InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError

try:
    response = baichuan_sdk.generate(model, messages)
except InvokeConnectionError as e:
    print(f"Connection error: {str(e)}")
except InvokeServerUnavailableError as e:
    print(f"Server unavailable: {str(e)}")
except InvokeRateLimitError as e:
    print(f"Rate limit exceeded: {str(e)}")
except InvokeAuthorizationError as e:
    print(f"Authorization error: {str(e)}")
except InvokeBadRequestError as e:
    print(f"Bad request: {str(e)}")
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Debugging

The SDK uses Python's logging module. To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will print detailed information about API requests and responses, which can be helpful for troubleshooting.

## Best Practices

1. Choose the appropriate model for your specific task (e.g., Baichuan2 models for advanced language tasks).
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Use environment variables for API keys in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. For embedding tasks, batch your inputs when possible to improve efficiency.
8. Regularly update the SDK to benefit from the latest features and bug fixes.

## Limitations

- The current SDK does not support model fine-tuning or training custom models.
- Some advanced features of the Baichuan API may not be directly accessible through this SDK. Refer to the official Baichuan API documentation for the most up-to-date information.

For more detailed information about available models, specific features, and API updates, please refer to the official Baichuan API documentation.
