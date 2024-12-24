class InvokeError(Exception):
    """Base class for all invoke errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class InvokeConnectionError(InvokeError):
    """Raised when there's a connection error during the API call."""
    pass

class InvokeServerUnavailableError(InvokeError):
    """Raised when the server is unavailable."""
    pass

class InvokeRateLimitError(InvokeError):
    """Raised when the API rate limit is exceeded."""
    pass

class InvokeAuthorizationError(InvokeError):
    """Raised when there's an authentication or authorization error."""
    pass

class InvokeBadRequestError(InvokeError):
    """Raised when the request is invalid or cannot be served."""
    pass

class InvokeTimeoutError(InvokeError):
    """Raised when the API request times out."""
    pass

class InvokeAPIError(InvokeError):
    """Raised for any other API-related errors not covered by the specific classes above."""
    pass

class InvokeModelNotFoundError(InvokeError):
    """Raised when the specified model is not found."""
    pass

class InvokeInvalidParameterError(InvokeError):
    """Raised when an invalid parameter is provided in the API call."""
    pass

def handle_anthropic_error(error: Exception) -> InvokeError:
    """
    Convert Anthropic-specific errors to our custom InvokeError types.
    """
    error_message = str(error)

    if isinstance(error, ConnectionError):
        return InvokeConnectionError(f"Connection error occurred: {error_message}")
    elif isinstance(error, TimeoutError):
        return InvokeTimeoutError(f"Request timed out: {error_message}")
    elif "rate limit" in error_message.lower():
        return InvokeRateLimitError(f"Rate limit exceeded: {error_message}")
    elif "unauthorized" in error_message.lower() or "forbidden" in error_message.lower():
        return InvokeAuthorizationError(f"Authorization error: {error_message}")
    elif "not found" in error_message.lower():
        if "model" in error_message.lower():
            return InvokeModelNotFoundError(f"Model not found: {error_message}")
        else:
            return InvokeBadRequestError(f"Resource not found: {error_message}")
    elif "bad request" in error_message.lower():
        return InvokeBadRequestError(f"Bad request: {error_message}")
    elif "server error" in error_message.lower():
        return InvokeServerUnavailableError(f"Server error: {error_message}")
    else:
        return InvokeAPIError(f"API error occurred: {error_message}")