# Cohere API Documentation

## Overview

Cohere provides access to its language models through an API. This document outlines how to use the Cohere API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Cohere API, initialize the OneSDK with your Cohere API key:

```python
from llm_onesdk import OneSDK

cohere_sdk = OneSDK("cohere", {"api_key": "your_api_key_here"})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "cohere-model-name"  # Replace with an actual Cohere model name
messages = [{"role": "user", "content": "Explain the concept of machine learning in simple terms."}]

response = cohere_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in cohere_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = cohere_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Creating Embeddings

Cohere supports creating embeddings for text:

```python
model = "embed-english-v2.0"  # Replace with the appropriate embedding model
input_text = "Hello, world!"

embeddings = cohere_sdk.create_embedding(model, input_text)
print(embeddings)
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = cohere_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.

For more detailed information about available models and specific features, please refer to the official Cohere API documentation.dd ge mdgem