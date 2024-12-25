
# Ollama API Documentation

## Overview

Ollama provides access to open-source language models that can be run locally. This document outlines how to use the Ollama API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Ollama API, initialize the OneSDK. Note that Ollama typically doesn't require an API key as it runs locally:

```python
from onesdk import OneSDK

ollama_sdk = OneSDK("ollama", {})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "llama2"  # Or another model you have pulled with Ollama
messages = [{"role": "user", "content": "Explain the concept of deep learning."}]

response = ollama_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in ollama_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = ollama_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Creating Embeddings

Ollama supports creating embeddings for text:

```python
model = "llama2"  # Or another model that supports embeddings
input_text = "Hello, world!"

embeddings = ollama_sdk.create_embedding(model, input_text)
print(embeddings)
```

### Listing Available Models

To see what models are available locally:

```python
models = ollama_sdk.list_models()
print(models)
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = ollama_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Ensure that Ollama is running locally before making API calls.
2. Use the most appropriate model for your task.
3. Implement proper error handling for robustness.
4. Be aware of the computational resources required to run models locally.

For more detailed information about available models and specific features, please refer to the official Ollama documentation.
