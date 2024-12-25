
# Kimi API Documentation

## Overview

Kimi (by Moonshot AI) provides access to its language models through an API. This document outlines how to use the Kimi API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Kimi API, initialize the OneSDK with your Kimi API key:

```python
from onesdk import OneSDK

kimi_sdk = OneSDK("kimi", {"api_key": "your_api_key_here"})
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
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = kimi_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Creating and Managing Context Caches

Kimi supports context caching for more efficient conversations:

```python
# Create a context cache
cache = kimi_sdk.create_cache(model, messages, name="My Context", description="A sample context")

# List available caches
caches = kimi_sdk.list_caches()

# Use a cache in a conversation
response = kimi_sdk.generate(model, messages, context_id=cache['id'])

# Update a cache
kimi_sdk.update_cache(cache['id'], metadata={"key": "value"})

# Delete a cache
kimi_sdk.delete_cache(cache['id'])
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = kimi_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Utilize context caching for more efficient and coherent conversations.

For more detailed information about available models and specific features, please refer to the official Kimi API documentation.

