
# OpenAI API Documentation

## Overview

OpenAI provides access to powerful language models through its API. This document outlines how to use the OpenAI API within the OneSDK framework.

## Basic Usage

### Initialization

To use the OpenAI API, initialize the OneSDK with your OpenAI API key:

```python
from llm_onesdk import OneSDK

openai_sdk = OneSDK("openai", {"api_key": "your_api_key_here"})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "gpt-3.5-turbo"  # Or another available OpenAI model
messages = [{"role": "user", "content": "Explain the concept of machine learning."}]

response = openai_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in openai_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = openai_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

### Creating Embeddings

OpenAI supports creating embeddings for text:

```python
model = "text-embedding-ada-002"  # OpenAI's embedding model
input_text = "Hello, world!"

embeddings = openai_sdk.create_embedding(model, input_text)
print(embeddings)
```

### Image Generation

OpenAI's DALL-E models can generate images from text descriptions:

```python
prompt = "A surrealist painting of a cat playing chess with a robot"
image_response = openai_sdk.create_image(prompt)
print(image_response)
```

### File Operations

OpenAI allows file uploads for certain use cases:

```python
# Upload a file
file_response = openai_sdk.upload_file("path/to/your/file.jsonl", purpose="fine-tune")

# List files
files = openai_sdk.list_files()

# Delete a file
openai_sdk.delete_file(file_response['id'])
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = openai_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Use the latest model versions for the best performance and capabilities.

For more detailed information about available models and specific features, please refer to the official OpenAI API documentation.
