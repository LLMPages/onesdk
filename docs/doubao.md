# Doubao API Documentation

## Overview

Doubao provides access to its advanced language models through an API. This document outlines how to use the Doubao API within the OneSDK framework, offering a seamless integration for various natural language processing tasks.

## Basic Usage

### Initialization

To use the Doubao API, initialize the OneSDK with your Doubao API key:

```python
from llm_onesdk import OneSDK

doubao_sdk = OneSDK("doubao", {
    "api_key": "your_api_key_here",
    "api_url": "https://ark.cn-beijing.volces.com/api/v3/"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key as an environment variable `DOUBAO_API_KEY`, and the SDK will automatically use it.

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "doubao-model-name"  # Replace with an actual Doubao model name
messages = [{"role": "user", "content": "What are the key principles of artificial intelligence?"}]

response = doubao_sdk.generate(model, messages, max_tokens=4096)
print(response['choices'][0]['message']['content'])
```

Additional parameters can be specified:

```python
response = doubao_sdk.generate(
    model,
    messages,
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
    stop=["END"],
    frequency_penalty=0.5,
    presence_penalty=0.5
)
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in doubao_sdk.stream_generate(model, messages, max_tokens=4096):
    print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = doubao_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Creating Embeddings

Doubao supports creating embeddings for text:

```python
model = "doubao-embedding-model"  # Replace with the appropriate embedding model
input_text = "Hello, world!"

embeddings = doubao_sdk.create_embedding(model, input_text, encoding_format="float")
print(embeddings)
```

### Tokenization

To tokenize text using a specific model:

```python
model = "doubao-tokenization-model"
text = "Your text here"
tokenization_result = doubao_sdk.tokenize(model, text)
print(tokenization_result)
```

### Context Creation and Usage

Doubao supports creating and using contexts for caching:

```python
# Create a context
context = doubao_sdk.create_context(model, messages, mode="session", ttl=86400)
context_id = context['context_id']

# Generate response using the context
response = doubao_sdk.generate_with_context(model, context_id, new_messages)
print(response['choices'][0]['message']['content'])
```

### Visual Understanding

For models supporting visual tasks:

```python
visual_messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]}
]
response = doubao_sdk.visual_generate(model, visual_messages)
print(response['choices'][0]['message']['content'])
```

## Advanced Features

### Setting a Proxy

To use a proxy for API calls:

```python
doubao_sdk.set_proxy("http://your-proxy-url:port")
```

## Error Handling

The SDK raises `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import (
    InvokeError, InvokeConnectionError, InvokeServerUnavailableError,
    InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError,
    InvokeUnsupportedOperationError
)

try:
    response = doubao_sdk.generate(model, messages)
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
except InvokeUnsupportedOperationError as e:
    print(f"Unsupported operation: {str(e)}")
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

1. Choose the appropriate model for your specific task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Use environment variables for API keys in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. Utilize context creation for session-based interactions to improve efficiency.
8. For visual tasks, ensure you're using models that support image input.
9. Regularly update the SDK to benefit from the latest features and bug fixes.

## Limitations

- The current SDK does not support listing models or getting individual model information.
- Some advanced features of the Doubao API may not be directly accessible through this SDK. Refer to the official Doubao API documentation for the most up-to-date information.

For more detailed information about available models, specific features, and API updates, please refer to the official Doubao API documentation.
