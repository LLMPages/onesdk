
# Qwen (通义千问) API Documentation

## Overview

Qwen, developed by Alibaba Cloud, provides access to powerful language models through its API. This document outlines how to use the Qwen API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Qwen API, initialize the OneSDK with your Qwen API key:

```python
from onesdk import OneSDK

qwen_sdk = OneSDK("qwen", {"api_key": "your_api_key_here"})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "qwen-turbo"  # Or another available Qwen model
messages = [{"role": "user", "content": "Explain the concept of artificial intelligence."}]

response = qwen_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
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

### Creating Embeddings

Qwen supports creating embeddings for text:

```python
model = "text-embedding-v1"  # Replace with the appropriate embedding model
input_text = "Hello, world!"

embeddings = qwen_sdk.create_embedding(model, input_text)
print(embeddings)
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = qwen_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.

For more detailed information about available models and specific features, please refer to the official Qwen API documentation.
