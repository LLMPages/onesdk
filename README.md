# OneSDK: Unified LLM API Interface

OneSDK is a Python library providing a unified interface for various Large Language Model (LLM) providers. It simplifies interactions with different LLM APIs through a consistent set of methods.

## Features

- Unified API for multiple LLM providers
- Flexible usage: per-call model specification or default model setting
- Intuitive interface for common LLM operations
- Synchronous and streaming text generation support
- Token counting functionality
- Embedding creation (for supported providers)
- Image generation (for supported providers)
- File operations (for supported providers)
- Proxy setting for API calls
- Usage statistics retrieval (for supported providers)

## Installation

```bash
pip install llm-onesdk
```

## Quick Start

OneSDK supports two main usage patterns:

### 1. Specify model for each call

```python
from llm_onesdk import OneSDK

sdk = OneSDK("openai", {"api_key": "your-api-key"})

response = sdk.generate(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Tell me a joke about programming."}]
)
print(response['choices'][0]['message']['content'])
```

### 2. Set a default model

```python
from llm_onesdk import OneSDK

sdk = OneSDK("openai", {"api_key": "your-api-key"})
sdk.set_model("gpt-3.5-turbo")

response = sdk.generate(
    messages=[{"role": "user", "content": "Tell me a joke about programming."}]
)
print(response['choices'][0]['message']['content'])
```

## Streaming Generation

```python
for chunk in sdk.stream_generate(
    model="gpt-3.5-turbo",  # Optional if using set_model()
    messages=[{"role": "user", "content": "Write a short story about AI."}]
):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

## Additional Operations

```python
# List models (for supported providers)
models = sdk.list_models()
print(models)

# Count tokens
token_count = sdk.count_tokens(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "How many tokens is this?"}]
)
print(f"Token count: {token_count}")

# Create embeddings (for supported providers)
embeddings = sdk.create_embedding(
    model="text-embedding-ada-002",
    input="Hello, world!"
)
print(embeddings)

# Generate image (for supported providers)
image_response = sdk.create_image("A futuristic city with flying cars")
print(image_response)
```

## Supported Providers and Core Methods

The following table shows the supported providers and their core method support:

| Provider  | list_models | generate | stream_generate | count_tokens | create_embedding | create_image |
|-----------|-------------|----------|-----------------|--------------|------------------|--------------|
| [Anthropic](docs/anthropic.md) | ✓           | ✓        | ✓               | ✓            | ✗                | ✗            |
| [Qwen (通义千问)](docs/qwen.md)      | ✓           | ✓        | ✓               | ✓            | ✓                | ✗            |
| [Cohere](docs/cohere.md)    | ✓           | ✓        | ✓               | ✓            | ✓                | ✗            |
| [Doubao](docs/doubao.md)    | ✓           | ✓        | ✓               | ✓            | ✓                | ✗            |
| [Gemini](docs/gemini.md)    | ✗           | ✓        | ✓               | ✓            | ✓                | ✗            |
| [Kimi](docs/kimi.md)      | ✓           | ✓        | ✓               | ✓            | ✗                | ✗            |
| [MiniMax](docs/minimax.md) | ✗           | ✓        | ✓               | ✓            | ✓                | ✓            |
| [Ollama](docs/ollama.md)    | ✓           | ✓        | ✓               | ✓            | ✓                | ✗            |
| [OpenAI](docs/openai.md)    | ✓           | ✓        | ✓               | ✓            | ✓                | ✓            |
| [Wenxin (文心一言)](docs/wenxin.md)    | ✗           | ✓        | ✓               | ✓            | ✗                | ✗            |

✓: Supported, ✗: Not supported

Note: Some providers may have additional provider-specific methods. Refer to individual provider documentation for details.

## Key Methods

- `set_model(model)`: Set default model
- `list_models()`: List available models (if supported)
- `generate(messages, model=None, **kwargs)`: Generate response
- `stream_generate(messages, model=None, **kwargs)`: Stream response
- `count_tokens(model, messages)`: Count tokens
- `create_embedding(model, input, **kwargs)`: Create embeddings (if supported)
- `create_image(prompt, **kwargs)`: Create image (if supported)
- `upload_file(file_path)`: Upload file (if supported)
- `set_proxy(proxy_url)`: Set proxy for API calls

## Error Handling

OneSDK uses custom exceptions inheriting from `InvokeError`. Always wrap API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import InvokeError

try:
    response = sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Documentation

For detailed information on each provider's capabilities and usage, please refer to the individual documentation files in the `docs/` directory.

## Contributing

We welcome contributions, especially new provider integrations! See our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is under the MIT License. See the [LICENSE](LICENSE) file for details.