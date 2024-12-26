# Qwen (通义千问) API Documentation

## Overview

Qwen, developed by Alibaba Cloud, provides access to powerful language models through its API. This document outlines how to use the Qwen API within the OneSDK framework, offering seamless integration for various AI tasks including text generation and multimodal interactions.

## Basic Usage

### Initialization

To use the Qwen API, initialize the OneSDK with your Qwen API key:

```python
from llm_onesdk import OneSDK

qwen_sdk = OneSDK("qwen", {
    "api_key": "your_api_key_here",
    "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key as an environment variable `DASHSCOPE_API_KEY`, and the SDK will automatically use it.

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "qwen-turbo"  # Or another available Qwen model
messages = [{"role": "user", "content": "Explain the concept of artificial intelligence."}]

response = qwen_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

#### Additional Parameters

You can customize the generation with additional parameters:

```python
response = qwen_sdk.generate(
    model,
    messages,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    max_tokens=100,
    stop=["END"],
    seed=42,
    enable_search=True
)
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in qwen_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = qwen_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

Note: This is a simple estimation based on word count and may not be fully accurate for all languages or special tokens.

## Advanced Features

### Multimodal Generation

Qwen supports multimodal models for tasks involving both text and images:

```python
model = "qwen-vl-plus"
messages = [
    {"role": "user", "content": [
        {"image": "base64_encoded_image_data"},
        {"text": "Describe this image."}
    ]}
]

response = qwen_sdk.generate(model, messages)
```

### System Messages

You can provide a system message to set the context or behavior of the model:

```python
response = qwen_sdk.generate(
    model,
    messages,
    system="You are a helpful assistant that specializes in technology."
)
```

### Tool Use

Qwen models can use tools when provided:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = qwen_sdk.generate(model, messages, tools=tools, tool_choice="auto")
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
    response = qwen_sdk.generate(model, messages)
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

1. Choose the appropriate model for your specific task (e.g., qwen-turbo for general tasks, qwen-vl-plus for multimodal tasks).
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Use environment variables for API keys in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. For multimodal tasks, ensure your images are properly encoded in base64 format.
8. When using tools, provide clear and detailed function descriptions for best results.
9. Regularly update the SDK to benefit from the latest features and bug fixes.
10. Use the token counting feature to estimate costs, but be aware it's an approximation.

## Setting a Proxy

To use a proxy for API calls:

```python
qwen_sdk.set_proxy("http://your-proxy-url:port")
```

## Limitations

- The current SDK does not support listing models or getting individual model information.
- Token counting is an estimation and may not be fully accurate for all use cases.
- Some features may be model-specific. Always refer to the official Qwen documentation for the most up-to-date information on model capabilities.

For more detailed information about available models, specific features, and API updates, please refer to the official Qwen API documentation by Alibaba Cloud.
