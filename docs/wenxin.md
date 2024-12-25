
# Wenxin (文心一言) API Documentation

## Overview

Wenxin, developed by Baidu, provides access to powerful language models through its API. This document outlines how to use the Wenxin API within the OneSDK framework.

## Basic Usage

### Initialization

To use the Wenxin API, initialize the OneSDK with your Wenxin API key and secret key:

```python
from llm_onesdk import OneSDK

wenxin_sdk = OneSDK("wenxin", {
    "api_key": "your_api_key_here",
    "secret_key": "your_secret_key_here"
})
```

### Generating Text

To generate text, use the `generate` method. Specify the model and provide a list of messages:

```python
model = "ERNIE-Bot"  # Or another available Wenxin model
messages = [{"role": "user", "content": "解释一下什么是机器学习。"}]

response = wenxin_sdk.generate(model, messages)
print(response['choices'][0]['message']['content'])
```

### Streaming Responses

For longer responses or to get partial results as they're generated, use the `stream_generate` method:

```python
for chunk in wenxin_sdk.stream_generate(model, messages):
    print(chunk['choices'][0]['message']['content'], end='', flush=True)
```

### Counting Tokens

To estimate the number of tokens in your input:

```python
token_count = wenxin_sdk.count_tokens(model, messages)
print(f"Token count: {token_count}")
```

## Error Handling

The SDK will raise `InvokeError` or its subclasses for various error conditions. Always wrap your API calls in try-except blocks:

```python
try:
    response = wenxin_sdk.generate(model, messages)
except InvokeError as e:
    print(f"An error occurred: {str(e)}")
```

## Best Practices

1. Use the most appropriate model for your task.
2. Implement proper error handling and retries for production applications.
3. Be mindful of rate limits and implement appropriate backoff strategies.
4. Keep your API key and secret key secure and never expose them in client-side code.
5. Consider the specific capabilities and limitations of each Wenxin model when designing your application.

For more detailed information about available models and specific features, please refer to the official Wenxin API documentation.
