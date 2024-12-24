# OneSDK: Unified LLM API Interface

OneSDK is a Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers. It simplifies the process of working with different LLM APIs by offering a consistent set of methods across providers.

## Features

- Unified API for multiple LLM providers
- Flexible usage patterns: specify model per call or set a default model
- Easy-to-use interface for common LLM operations
- Support for both synchronous and streaming text generation
- Token counting functionality
- File upload capability (where supported)
- Proxy setting for API calls
- Usage statistics retrieval (where available)

## Installation

```bash
pip install llm-onesdk
```

## Quick Start

OneSDK supports two main usage patterns:

### Pattern 1: Specify model for each call

```python
from llm-onesdk import OneSDK

# Initialize the SDK with your chosen provider and credentials
sdk = OneSDK("anthropic", {"api_key": "your-api-key"})

# Generate text
response = sdk.generate(
    model="claude-2",
    messages=[{"role": "user", "content": "Tell me a joke about programming."}]
)
print(response['choices'][0]['message']['content'])
```

### Pattern 2: Set a default model

```python
from onesdk import OneSDK

# Initialize the SDK and set a default model
sdk = OneSDK("anthropic", {"api_key": "your-api-key"})
sdk.set_model("claude-2")

# Generate text using the default model
response = sdk.generate(
    messages=[{"role": "user", "content": "Tell me a joke about programming."}]
)
print(response['choices'][0]['message']['content'])
```

## Streaming Generation

Both patterns support streaming generation:

```python
for chunk in sdk.stream_generate(
    model="claude-2",  # or omit if using set_model()
    messages=[{"role": "user", "content": "Write a short story about AI."}]
):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

## Other Operations

```python
# List available models
models = sdk.list_models()
print(models)

# Count tokens
token_count = sdk.count_tokens(
    model="claude-2",
    messages=[{"role": "user", "content": "How many tokens is this?"}]
)
print(f"Token count: {token_count}")
```

## Supported Providers

You can list all available providers using:

```python
print(OneSDK.list_providers())
```

## Methods

- `set_model(model)`: Set the default model for subsequent API calls.
- `list_models()`: List available models for the current provider.
- `get_model_info(model_id)`: Get information about a specific model.
- `generate(messages, model=None, **kwargs)`: Generate a response using the specified or default model.
- `stream_generate(messages, model=None, **kwargs)`: Generate a streaming response.
- `count_tokens(model, messages)`: Count the number of tokens in the input messages.
- `create_completion(model, prompt, **kwargs)`: Create a completion using the legacy API (if supported).
- `upload_file(file_path)`: Upload a file and return a reference.
- `set_proxy(proxy_url)`: Set a proxy for API calls.
- `get_usage()`: Get usage statistics for the current account.

## Error Handling

OneSDK uses custom exception classes for error handling. The base exception class is `InvokeError`, with specific subclasses for different error types (e.g., `InvokeModelNotFoundError`).

## Contributing

We welcome contributions, especially new provider integrations! Please see our [Contributing Guide](CONTRIBUTING.md) for more information on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.