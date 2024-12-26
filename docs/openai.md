# OpenAI API Documentation

## Overview

OpenAI provides access to powerful language models and various AI capabilities through its API. This document outlines how to use the OpenAI API within the OneSDK framework, offering a seamless integration for a wide range of AI tasks including text generation, embeddings, image generation, audio processing, and fine-tuning.

## Basic Usage

### Initialization

To use the OpenAI API, initialize the OneSDK with your OpenAI API key:

```python
from llm_onesdk import OneSDK

openai_sdk = OneSDK("openai", {
    "api_key": "your_api_key_here",
    "api_url": "https://api.openai.com/"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key as an environment variable `OPENAI_API_KEY`, and the SDK will automatically use it.

### Listing Available Models

To get a list of available models:

```python
models = openai_sdk.list_models()
print(models)
```

### Getting Model Information

To get information about a specific model:

```python
model_info = openai_sdk.get_model("gpt-3.5-turbo")
print(model_info)
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
    print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

### Creating Embeddings

OpenAI supports creating embeddings for text:

```python
model = "text-embedding-ada-002"  # OpenAI's embedding model
input_text = "Hello, world!"

embeddings = openai_sdk.create_embedding(model, input_text)
print(embeddings)
```

## Advanced Features

### Image Generation and Manipulation

OpenAI's DALL-E models can generate and manipulate images:

```python
# Generate an image
prompt = "A surrealist painting of a cat playing chess with a robot"
image_response = openai_sdk.create_image(prompt, n=1, size="1024x1024")

# Edit an image
with open("image.png", "rb") as image_file, open("mask.png", "rb") as mask_file:
    edit_response = openai_sdk.create_edit(image_file, mask_file, "Add a hat to the person")

# Create image variations
with open("image.png", "rb") as image_file:
    variation_response = openai_sdk.create_variation(image_file, n=3)
```

### Audio Processing

OpenAI provides audio transcription and translation capabilities:

```python
# Transcribe audio
with open("audio.mp3", "rb") as audio_file:
    transcription = openai_sdk.create_transcription(audio_file, "whisper-1")

# Translate audio
with open("foreign_audio.mp3", "rb") as audio_file:
    translation = openai_sdk.create_translation(audio_file, "whisper-1")

# Generate speech
speech_audio = openai_sdk.create_speech("tts-1", "Hello, world!", "alloy")
with open("speech.mp3", "wb") as audio_file:
    audio_file.write(speech_audio)
```

### Content Moderation

Use OpenAI's content moderation:

```python
moderation_result = openai_sdk.create_moderation("Text to be moderated")
print(moderation_result)
```

### File Operations

OpenAI allows file uploads for certain use cases:

```python
# Upload a file
with open("data.jsonl", "rb") as file:
    file_response = openai_sdk.upload_file(file, purpose="fine-tune")

# List files
files = openai_sdk.list_files()

# Get file info
file_info = openai_sdk.get_file_info(file_response['id'])

# Get file content
file_content = openai_sdk.get_file_content(file_response['id'])

# Delete a file
openai_sdk.delete_file(file_response['id'])
```

### Fine-Tuning

Create and manage fine-tuning jobs:

```python
# Create a fine-tuning job
job = openai_sdk.create_fine_tuning_job(training_file="file-abc123", model="gpt-3.5-turbo")

# List fine-tuning jobs
jobs = openai_sdk.list_fine_tuning_jobs()

# Get fine-tuning job info
job_info = openai_sdk.get_fine_tuning_job(job['id'])

# Cancel a fine-tuning job
openai_sdk.cancel_fine_tuning_job(job['id'])

# List fine-tuning events
events = openai_sdk.list_fine_tuning_events(job['id'])
```

## Error Handling

The SDK raises `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import (
    InvokeError, InvokeConnectionError, InvokeServerUnavailableError,
    InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError
)

try:
    response = openai_sdk.generate(model, messages)
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

1. Choose the appropriate model for your specific task (e.g., GPT-4 for complex reasoning, Ada for simpler tasks).
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key secure and never expose it in client-side code.
5. Use environment variables for API keys in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. For file operations, ensure you're using the correct 'purpose' parameter.
8. When fine-tuning models, carefully prepare your training data for best results.
9. For image and audio tasks, pay attention to file format and size requirements.
10. Regularly update the SDK to benefit from the latest features and bug fixes.

## Setting a Proxy

To use a proxy for API calls:

```python
openai_sdk.set_proxy("http://your-proxy-url:port")
```

For more detailed information about available models, specific features, and API updates, please refer to the official OpenAI API documentation.
