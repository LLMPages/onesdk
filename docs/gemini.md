# Gemini API Documentation

## Overview

Google's Gemini provides access to powerful language models through its API. This document outlines how to use the Gemini API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Gemini API, initialize the OneSDK with your Gemini API key:

```python
from llm_onesdk import OneSDK

gemini_sdk = OneSDK("gemini", {"api_key": "your_api_key_here"})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "gemini-pro"  # Or another available Gemini model
messages = [{"role": "user", "content": "Explain the concept of neural networks in simple terms."}]

response = gemini_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in gemini_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = gemini_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Handling Multimodal Inputs

Gemini models support multimodal inputs, including images:

```python
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "base64_encoded_image_data"}}
    ]}
]

response = gemini_sdk.generate("gemini-pro-vision", messages)
print(response['choices'][0]['message']['content'])
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = gemini_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.

For more detailed information about available models and specific features, please refer to the official Gemini API documentation.
mimi o