c# Anthropic API Documentation

## Overview

Anthropic provides access to powerful language models through their API. This document outlines how to use the Anthropic API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Anthropic API, you first need to initialize the OneSDK with your Anthropic API key:

```python
from onesdk import OneSDK

anthropic_sdk = OneSDK("anthropic", {"api_key": "your_api_key_here"})
```

### Generating Text

To generate text, use the `generate` method. You need to specify the model and provide a list of messages:

```python
model = "claude-3-opus-20240229"
messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]

response = anthropic_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in anthropic_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = anthropic_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

## Advanced Features

### Setting a Default Model

You can set a default model to use for all subsequent calls:

```python
anthropic_sdk.set_model("claude-3-opus-20240229")
response = anthropic_sdk.generate(messages=messages)  # No need to specify model
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = anthropic_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.

For more detailed information about available models and specific features, please refer to the official Anthropic API documentation.
