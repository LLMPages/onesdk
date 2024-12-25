from typing import Optional, Dict, Any

class InvokeError(Exception):
    """Base class for all invoke errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 http_status: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        error_parts = [f"Error {self.error_code}: {self.message}" if self.error_code else self.message]
        if self.http_status:
            error_parts.append(f"HTTP Status: {self.http_status}")
        if self.details:
            error_parts.append(f"Details: {self.details}")
        return " | ".join(error_parts)

class InvokeConnectionError(InvokeError):
    """Raised when there's a connection error during the API call."""
    pass

class InvokeServerUnavailableError(InvokeError):
    """Raised when the server is unavailable."""
    pass

class InvokeRateLimitError(InvokeError):
    """Raised when the API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

    def __str__(self):
        base_str = super().__str__()
        if self.retry_after:
            return f"{base_str} | Retry after: {self.retry_after} seconds"
        return base_str

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

def handle_api_error(error: Exception, provider: str) -> InvokeError:
    """
    Convert provider-specific errors to our custom InvokeError types.
    """
    error_message = str(error)
    error_type = type(error).__name__
    http_status = getattr(error, 'status_code', None) if hasattr(error, 'status_code') else None

    # Common error handling
    if isinstance(error, ConnectionError):
        return InvokeConnectionError(f"Connection error occurred: {error_message}", 
                                     error_code="CONNECTION_ERROR", http_status=http_status)
    elif isinstance(error, TimeoutError):
        return InvokeTimeoutError(f"Request timed out: {error_message}", 
                                  error_code="TIMEOUT", http_status=http_status)

    # Provider-specific error handling
    if provider.lower() == "anthropic":
        return _handle_anthropic_error(error_message, error_type, http_status)
    elif provider.lower() == "openai":
        return _handle_openai_error(error_message, error_type, http_status)
    # Add more providers as needed
    else:
        return InvokeAPIError(f"Unhandled {provider} API error occurred: {error_message}", 
                              error_code="UNKNOWN_ERROR", http_status=http_status)

def _handle_anthropic_error(error_message: str, error_type: str, http_status: Optional[int]) -> InvokeError:
    if "rate limit" in error_message.lower():
        return InvokeRateLimitError(f"Rate limit exceeded: {error_message}", 
                                    error_code="RATE_LIMIT_EXCEEDED", http_status=http_status)
    elif "unauthorized" in error_message.lower() or "forbidden" in error_message.lower():
        return InvokeAuthorizationError(f"Authorization error: {error_message}", 
                                        error_code="UNAUTHORIZED", http_status=http_status)
    elif "not found" in error_message.lower():
        if "model" in error_message.lower():
            return InvokeModelNotFoundError(f"Model not found: {error_message}", 
                                            error_code="MODEL_NOT_FOUND", http_status=http_status)
        else:
            return InvokeBadRequestError(f"Resource not found: {error_message}", 
                                         error_code="RESOURCE_NOT_FOUND", http_status=http_status)
    elif "bad request" in error_message.lower():
        return InvokeBadRequestError(f"Bad request: {error_message}", 
                                     error_code="BAD_REQUEST", http_status=http_status)
    elif "server error" in error_message.lower():
        return InvokeServerUnavailableError(f"Server error: {error_message}", 
                                            error_code="SERVER_ERROR", http_status=http_status)
    else:
        return InvokeAPIError(f"API error occurred: {error_message}", 
                              error_code="UNKNOWN_ERROR", http_status=http_status)

def _handle_openai_error(error_message: str, error_type: str, http_status: Optional[int]) -> InvokeError:
    # Add OpenAI specific error handling here
    # This is just an example, adjust according to actual OpenAI error patterns
    if "rate limit" in error_message.lower():
        return InvokeRateLimitError(f"Rate limit exceeded: {error_message}", 
                                    error_code="RATE_LIMIT_EXCEEDED", http_status=http_status)
    # Add more OpenAI specific error handling...
    return InvokeAPIError(f"OpenAI API error occurred: {error_message}", 
                          error_code="UNKNOWN_ERROR", http_status=http_status)