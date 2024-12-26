# Anthropic API Documentation

## Overview

Anthropic provides access to powerful language models through their API. This document outlines how to use the Anthropic API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Anthropic API, you first need to initialize the OneSDK with your Anthropic API key:

```python
from llm_onesdk import OneSDK

anthropic_sdk = OneSDK("anthropic", {
    "api_key": "your_api_key_here",
    "api_url": "https://api.anthropic.com"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key as an environment variable `ANTHROPIC_API_KEY`, and the SDK will automatically use it.

### Listing Available Models

To get a list of available models:

```python
models = anthropic_sdk.list_models()
print(models)
```

### Getting Model Information

To get information about a specific model:

```python
model_info = anthropic_sdk.get_model("claude-3-opus-20240229")
print(model_info)
```

### Generating Text

To generate text, use the `generate` method. You need to specify the model and provide a list of messages:

```python
model = "claude-3-opus-20240229"
messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]

response = anthropic_sdk.generate(model, messages, max_tokens=1000)
print(response['content'][0]['text'])
```

#### Additional Parameters

You can customize the generation with additional parameters:

```python
response = anthropic_sdk.generate(
    model,
    messages,
    max_tokens=1000,
    temperature=0.7,
    top_p=0.9,
    stop=["END"],
    metadata={"user_id": "12345"}
)
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in anthropic_sdk.stream_generate(model, messages, max_tokens=1000):
    print(chunk['delta']['text'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = anthropic_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

## Advanced Features

### Setting a Proxy

To use a proxy for API calls:

```python
anthropic_sdk.set_proxy("http://your-proxy-url:port")
```

### Handling Images

The API supports sending images as part of the message content. To include an image:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image",
                "source": {
                    "type": "path",
                    "path": "/path/to/your/image.jpg",
                    "media_type": "image/jpeg"
                }
            }
        ]
    }
]

response = anthropic_sdk.generate(model, messages)
```

### Changing the API Base URL

If you need to use a different API endpoint (e.g., for testing or using a proxy service), you can specify a custom base URL during initialization:

```python
anthropic_sdk = OneSDK("anthropic", {
    "api_key": "your_api_key_here",
    "api_url": "https://your-custom-endpoint.com"
})
```

### Handling Different API Versions

The SDK uses API version "2023-06-01" by default. If you need to use a different version, you can modify the `API_VERSION` class variable in the `API` class:

```python
from llm_onesdk.providers.anthropic.api import API

API.API_VERSION = "your-desired-version"
```

Note: Changing the API version may affect the behavior and available features. Consult the official Anthropic API documentation for version-specific information.

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import InvokeError, InvokeConnectionError, InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError

try:
    response = anthropic_sdk.generate(model, messages)
except InvokeConnectionError as e:
    print(f"Connection error: {str(e)}")
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

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. When working with large responses, use the streaming API to improve responsiveness.
6. Set a reasonable `max_tokens` value to control the length of generated responses.
7. Use the `count_tokens` method to estimate costs and stay within token limits.
8. When processing sensitive data, ensure you're complying with data protection regulations.
9. Regularly update the SDK to benefit from the latest features and bug fixes.
10. Use environment variables for API keys in production environments.

## Limitations

- The SDK currently does not support fine-tuning or training custom models.
- Some advanced features of the Anthropic API may not be directly accessible through this SDK. Refer to the official Anthropic API documentation for the most up-to-date information.

## Note on API Versions

This SDK uses the Anthropic API version "2023-06-01" by default. Make sure your use cases are compatible with this version, or change it as described in the "Handling Different API Versions" section.

For more detailed information about available models, specific features, and API updates, please refer to the official Anthropic API documentation.
