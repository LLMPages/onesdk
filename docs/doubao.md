# Doubao API Documentation

## Overview

Doubao provides access to its language models through an API. This document outlines how to use the Doubao API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Doubao API, initialize the OneSDK with your Doubao API key:

```python
from llm_onesdk import OneSDK

doubao_sdk = OneSDK("doubao", {"api_key": "your_api_key_here"})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "doubao-model-name"  # Replace with an actual Doubao model name
messages = [{"role": "user", "content": "What are the key principles of artificial intelligence?"}]

response = doubao_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in doubao_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
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

embeddings = doubao_sdk.create_embedding(model, input_text)
print(embeddings)
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = doubao_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.

For more detailed information about available models and specific features, please refer to the official Doubao API documentation.
