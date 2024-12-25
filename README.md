# OneSDK: Unified LLM API Interface

OneSDK is a Python library providing a unified interface for various Large Language Model (LLM) providers. It simplifies interactions with different LLM APIs through a consistent set of methods.

## Features

- Unified API for multiple LLM providers
- Flexible usage: per-call model specification or default model setting
- Intuitive interface for common LLM operations
- Synchronous and streaming text generation support
- Token counting functionality
- File upload capability (provider-dependent)
- Proxy setting for API calls
- Usage statistics retrieval (provider-dependent)

## Installation

```bash
pip install llm-onesdk
```

## Quick Start

OneSDK supports two main usage patterns:

### 1. Specify model for each call

```python
from llm_onesdk import OneSDK

sdk = OneSDK("anthropic", {"api_key": "your-api-key"})

response = sdk.generate(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Tell me a joke about programming."}]
)
print(response['choices'][0]['message']['content'])
```

### 2. Set a default model

```python
from llm_onesdk import OneSDK

sdk = OneSDK("anthropic", {"api_key": "your-api-key"})
sdk.set_model("claude-3-opus-20240229")

response = sdk.generate(
    messages=[{"role": "user", "content": "Tell me a joke about programming."}]
)
print(response['choices'][0]['message']['content'])
```

## Streaming Generation

```python
for chunk in sdk.stream_generate(
    model="claude-3-opus-20240229",  # Optional if using set_model()
    messages=[{"role": "user", "content": "Write a short story about AI."}]
):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

## Additional Operations

```python
# List models
models = sdk.list_models()
print(models)

# Count tokens
token_count = sdk.count_tokens(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "How many tokens is this?"}]
)
print(f"Token count: {token_count}")
```

## Supported Providers and Methods

The following table shows the supported providers, their default models, and the methods they support:

| Provider  | list_models | generate | stream_generate | count_tokens | create_embedding | create_image |
|-----------|-------------|----------|-----------------|--------------|------------------|--------------|
| Anthropic | ✓           | ✓        | ✓               | ✓            | ✗                | ✗            |
| Qwen      | ✓           | ✓        | ✓               | ✓            | ✓                | ✗            |
| Cohere    | ✗           | ✓        | ✓               | ✓            | ✓                | ✗            |
| Doubao    | ✗           | ✓        | ✓               | ✓            | ✓                | ✗            |
| Gemini    | ✗           | ✓        | ✓               | ✗            | ✗                | ✗            |
| Minimax   | ✗           | ✓        | ✓               | ✓            | ✓                | ✓            |
| OpenAI    | ✓           | ✓        | ✓               | ✓            | ✓                | ✓            |
| Wenxin    | ✗           | ✓        | ✓               | ✓            | ✗                | ✗            |

✓: Supported, ✗: Not supported

## Key Methods

- `set_model(model)`: Set default model
- `list_models()`: List available models
- `get_model_info(model_id)`: Get model information
- `generate(messages, model=None, **kwargs)`: Generate response
- `stream_generate(messages, model=None, **kwargs)`: Stream response
- `count_tokens(model, messages)`: Count tokens
- `create_completion(model, prompt, **kwargs)`: Legacy API completion
- `create_embedding(model, input, **kwargs)`: Create embeddings
- `create_image(prompt, **kwargs)`: Create image (for supported providers)
- `upload_file(file_path)`: Upload file
- `set_proxy(proxy_url)`: Set proxy
- `get_usage()`: Get usage statistics

## Error Handling

OneSDK uses custom exceptions inheriting from `InvokeError` (e.g., `InvokeModelNotFoundError`).

## Contributing

We welcome contributions, especially new provider integrations! See our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is under the MIT License. See the [LICENSE](LICENSE) file for details.