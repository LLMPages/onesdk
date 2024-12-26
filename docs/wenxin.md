# Wenxin (文心一言) API Documentation

## Overview

Wenxin, developed by Baidu, provides access to powerful language models through its API. This document outlines how to use the Wenxin API within the OneSDK framework, offering seamless integration for various AI tasks including text generation and conversational AI.

## Basic Usage

### Initialization

To use the Wenxin API, initialize the OneSDK with your Wenxin API key and secret key:

```python
from llm_onesdk import OneSDK

wenxin_sdk = OneSDK("wenxin", {
    "api_key": "your_api_key_here",
    "secret_key": "your_secret_key_here",
    "api_url": "https://aip.baidubce.com"  # Optional: Use this to override the default base URL
})
```

Alternatively, you can set the API key and secret key as environment variables `WENXIN_API_KEY` and `WENXIN_SECRET_KEY`, and the SDK will automatically use them.

### Available Models

The SDK supports the following models by default:
- ERNIE-Bot
- ERNIE-Bot-turbo
- BLOOMZ-7B

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "ERNIE-Bot"  # Or another available Wenxin model
messages = [{"role": "user", "content": "解释一下什么是机器学习。"}]

response = wenxin_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

#### Additional Parameters

You can customize the generation with additional parameters:

```python
response = wenxin_sdk.generate(
    model,
    messages,
    temperature=0.7,
    top_p=0.9,
    penalty_score=1.0,
    user_id="unique_user_id"
)
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in wenxin_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = wenxin_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

Note: This is a simple estimation based on word count and may not be fully accurate for all languages or special tokens.

## Advanced Features

### Setting Custom Models

You can set custom models and their corresponding endpoints:

```python
wenxin_sdk.set_custom_model("CustomModel", "/custom/model/endpoint")
```

## Error Handling

The SDK raises `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
from llm_onesdk.utils.error_handler import (
    InvokeError, InvokeConnectionError, InvokeRateLimitError,
    InvokeAuthorizationError, InvokeBadRequestError
)

try:
    response = wenxin_sdk.generate(model, messages)
except InvokeConnectionError as e:
    print(f"Connection error: {str(e)}")
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

1. Choose the appropriate model for your specific task (e.g., ERNIE-Bot for complex tasks, ERNIE-Bot-turbo for faster responses).
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key and secret key secure and never expose them in client-side code.
5. Use environment variables for API keys and secret keys in production environments.
6. When working with large responses, use the streaming API to improve responsiveness.
7. Consider the specific capabilities and limitations of each Wenxin model when designing your application.
8. Use the token counting feature to estimate costs, but be aware it's an approximation.
9. Regularly update the SDK to benefit from the latest features and bug fixes.
10. For multi-turn conversations, properly structure your messages with appropriate role assignments (e.g., "user", "assistant").

## Setting a Proxy

To use a proxy for API calls:

```python
wenxin_sdk.set_proxy("http://your-proxy-url:port")
```

## Limitations

- Token counting is an estimation and may not be fully accurate for all use cases.
- The SDK currently does not support advanced features like fine-tuning or training custom models.
- Some features may be model-specific. Always refer to the official Wenxin documentation for the most up-to-date information on model capabilities.

For more detailed information about available models, specific features, and API updates, please refer to the official Wenxin API documentation by Baidu.
