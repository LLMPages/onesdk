# MiniMax API Documentation

## Overview

MiniMax provides access to its language models through an API. This document outlines how to use the MiniMax API within the OneSDK framework.

## Basic Usage

### Initialization

To use the MiniMax API, initialize the OneSDK with your MiniMax API key and group ID:

```python
from llm_onesdk import OneSDK

minimax_sdk = OneSDK("minimax", {
    "api_key": "your_api_key_here",
    "group_id": "your_group_id_here"
})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "minimax-model-name"  # Replace with an actual MiniMax model name
messages = [{"role": "user", "content": "What are the main applications of machine learning?"}]

response = minimax_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in minimax_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = minimax_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Creating Embeddings

MiniMax supports creating embeddings for text:

```python
model = "minimax-embedding-model"  # Replace with the appropriate embedding model
input_text = "Hello, world!"

embeddings = minimax_sdk.create_embedding(model, input_text)
print(embeddings)
```

### Text-to-Speech

MiniMax offers text-to-speech capabilities:

```python
text = "Hello, this is a test of text-to-speech functionality."
voice_id = "voice_id_here"  # Replace with an actual voice ID

audio_data = minimax_sdk.text_to_speech(text, voice_id)
# Save or process the audio_data as needed
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = minimax_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key and group ID secure and never expose them in client-side code.

For more detailed information about available models and specific features, please refer to the official MiniMax API documentation.
