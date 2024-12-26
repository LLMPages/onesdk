# Kimi API Documentation

## Overview

Kimi (by Moonshot AI) provides access to its advanced language models through an API. This document outlines how to use the Kimi API within the OneSDK framework, offering a seamless integration for various natural language processing tasks.

## Basic Usage

### Initialization

To use the Kimi API, initialize the OneSDK with your Kimi API key:

```python
from llm_onesdk import OneSDK

kimi_sdk = OneSDK("kimi", {
    "api_key": "your_api_key_here",
    "api_url": "https://api.moonshot.cn/v1/"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key as an environment variable `MOONSHOT_API_KEY`, and the SDK will automatically use it.

### Listing Available Models

To get a list of available models:

```python
models = kimi_sdk.list_models()
print(models)
```

### Getting Model Information

To get information about a specific model:

```python
model_info = kimi_sdk.get_model("model_id")
print(model_info)
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "kimi-model-name"  # Replace with an actual Kimi model name
messages = [{"role": "user", "content": "Explain the concept of blockchain technology."}]

response = kimi_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in kimi_sdk.stream_generate(model, messages):
    print(chunk['delta']['text'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = kimi_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

## Advanced Features

### Creating and Managing Context Caches

Kimi supports context caching for more efficient conversations:

```python
# Create a context cache
cache = kimi_sdk.create_cache(
    model, 
    messages, 
    tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a location"}}],
    name="My Context", 
    description="A sample context",
    metadata={"key": "value"},
    ttl=3600  # Cache expires after 1 hour
)

# List available caches
caches = kimi_sdk.list_caches(limit=10, order="desc", metadata={"key": "value"})

# Get information about a specific cache
cache_info = kimi_sdk.get_cache(cache['id'])

# Update a cache
kimi_sdk.update_cache(cache['id'], metadata={"new_key": "new_value"}, ttl=7200)

# Delete a cache
kimi_sdk.delete_cache(cache['id'])
```

### Managing Tags

Kimi allows you to create and manage tags for context caches:

```python
# Create a tag for a cache
tag = kimi_sdk.create_tag("my_tag", cache['id'])

# List tags
tags = kimi_sdk.list_tags(limit=10, order="desc")

# Get information about a specific tag
tag_info = kimi_sdk.get_tag("my_tag")

# Get the content of a cache associated with a tag
tag_content = kimi_sdk.get_tag_content("my_tag")

# Delete a tag
kimi_sdk.delete_tag("my_tag")
```

### Setting a Proxy

To use a proxy for API calls:

```python
kimi_sdk.set_proxy("http://your-proxy-url:port")
```

## Error Handling

The SDK raises `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import (
    InvokeError, InvokeConnectionError, InvokeServerUnavailableError,
    InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError
)

try:
    response = kimi_sdk.generate(model, messages)
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

1. Choose the appropriate model for your specific task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Use environment variables for API keys in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. Utilize context caching for more efficient and coherent conversations.
8. Use tags to organize and quickly access specific context caches.
9. Regularly update the SDK to benefit from the latest features and bug fixes.
10. Monitor your token usage to optimize costs and performance.

For more detailed information about available models, specific features, and API updates, please refer to the official Kimi API documentation by Moonshot AI.
