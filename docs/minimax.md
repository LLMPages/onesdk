# MiniMax API Documentation

## Overview

MiniMax provides access to its advanced language models and various AI capabilities through an API. This document outlines how to use the MiniMax API within the OneSDK framework, offering a seamless integration for a wide range of AI tasks including text generation, embeddings, text-to-speech, video generation, music generation, and knowledge base management.

## Basic Usage

### Initialization

To use the MiniMax API, initialize the OneSDK with your MiniMax API key and group ID:

```python
from llm_onesdk import OneSDK

minimax_sdk = OneSDK("minimax", {
    "api_key": "your_api_key_here",
    "group_id": "your_group_id_here",
    "api_url": "https://api.minimax.chat/v1/"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key and group ID as environment variables `MINIMAX_API_KEY` and `MINIMAX_GROUP_ID`, and the SDK will automatically use them.

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "abab5.5-chat"  # Replace with an actual MiniMax model name
messages = [{"role": "user", "content": "What are the main applications of machine learning?"}]

response = minimax_sdk.generate(model, messages, tokens_to_generate=2048, temperature=0.7, top_p=0.95)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in minimax_sdk.stream_generate(model, messages, tokens_to_generate=2048):
    print(chunk['delta']['text'], end='', flush=True)
```

### Creating Embeddings

MiniMax supports creating embeddings for text:

```python
model = "embo-01"  # Replace with the appropriate embedding model
texts = ["Hello, world!", "Another example text"]
type = "document"  # or "query"

embeddings = minimax_sdk.create_embedding(model, texts, type)
print(embeddings)
```

## Advanced Features

### Text-to-Speech

MiniMax offers text-to-speech capabilities:

```python
model = "speech-01"
text = "Hello, this is a test of text-to-speech functionality."
voice_setting = {"name": "zh_female_qingxin"}
audio_setting = {"speed": 1.0, "volume": 1.0}

response = minimax_sdk.text_to_speech(model, text, voice_setting, audio_setting)
# Process the audio data in the response
```

### Video Generation

Create and query video generation tasks:

```python
model = "video-01"
prompt = "A serene lake surrounded by mountains at sunset"

task = minimax_sdk.create_video_generation_task(model, prompt)
task_id = task['task_id']

# Query the task status
status = minimax_sdk.query_video_generation_task(task_id)
print(status)
```

### Music Generation

Upload music files and generate music based on lyrics:

```python
# Upload a music file
with open('music.mp3', 'rb') as file:
    upload_response = minimax_sdk.upload_music(file, purpose='song')

# Generate music
model = "music-01"
lyrics = "Your lyrics here"
refer_voice = upload_response['file_id']  # Optional
refer_instrumental = None  # Optional

music_response = minimax_sdk.generate_music(model, lyrics, refer_voice, refer_instrumental)
print(music_response)
```

### File Management

MiniMax allows you to manage files for various purposes:

```python
# List files
files = minimax_sdk.list_files(purpose='song')

# Upload a file
with open('document.pdf', 'rb') as file:
    upload_response = minimax_sdk.upload_file(file, purpose='knowledge_base')

# Get file info
file_info = minimax_sdk.get_file_info(upload_response['file_id'])

# Delete a file
minimax_sdk.delete_file(upload_response['file_id'])
```

### Knowledge Base Management

Create and manage knowledge bases for enhanced AI capabilities:

```python
# Create a knowledge base
kb_response = minimax_sdk.create_knowledge_base("My Knowledge Base", "embo-01", operator_id=12345)

# Add a document to the knowledge base
minimax_sdk.add_document_to_knowledge_base(kb_response['knowledge_base_id'], file_id, operator_id=12345)

# List knowledge bases
kb_list = minimax_sdk.list_knowledge_bases()

# Delete a knowledge base
minimax_sdk.delete_knowledge_base(kb_response['knowledge_base_id'], operator_id=12345)
```

### ChatCompletion Pro

Use the advanced ChatCompletion Pro API for more controlled conversations:

```python
response = minimax_sdk.chatcompletion_pro(
    model,
    messages,
    bot_setting=[{"bot_name": "CustomBot", "content": "This is a custom AI assistant."}],
    reply_constraints={"sender_type": "BOT", "sender_name": "CustomBot"}
)
print(response['choices'][0]['message']['content'])
```

### Voice Cloning and Generation

Clone voices and generate voices based on text descriptions:

```python
# Clone a voice
cloned_voice = minimax_sdk.voice_cloning(file_id, voice_id)

# Generate a voice based on description
generated_voice = minimax_sdk.text_to_voice("male", "adult", ["deep", "calm"], "Hello, this is a test.")

# Delete a voice
minimax_sdk.delete_voice("cloned", voice_id)
```

## Error Handling

The SDK raises `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import (
    InvokeError, InvokeConnectionError, InvokeServerUnavailableError,
    InvokeRateLimitError, InvokeAuthorizationError, InvokeBadRequestError
)

try:
    response = minimax_sdk.generate(model, messages)
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

1. Choose the appropriate model for your specific task (e.g., abab5.5-chat for text generation, embo-01 for embeddings).
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key and group ID secure and never expose them in client-side code.
5. Use environment variables for API keys and group IDs in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. For file uploads, ensure you're using the correct 'purpose' parameter.
8. When creating knowledge bases, choose an appropriate embedding model.
9. For voice and music generation tasks, pay attention to the specific parameters required (e.g., voice settings, audio settings).
10. Regularly update the SDK to benefit from the latest features and bug fixes.

## Setting a Proxy

To use a proxy for API calls:

```python
minimax_sdk.set_proxy("http://your-proxy-url:port")
```

For more detailed information about available models, specific features, and API updates, please refer to the official MiniMax API documentation.
