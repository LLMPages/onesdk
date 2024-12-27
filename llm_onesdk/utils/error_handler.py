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

class InvokeUnsupportedOperationError(InvokeError):
    """Raised when an unsupported operation is attempted."""
    pass

class InvokeConfigError(InvokeError):
    """Raised when there's a configuration error."""
    pass

def handle_api_error(error: Exception) -> InvokeError:
    """
    Convert provider-specific errors to our custom InvokeError types.
    This function should be implemented in each provider's specific API module.
    """
    # Default error handling
    error_message = str(error)
    http_status = getattr(error, 'status_code', None) if hasattr(error, 'status_code') else None

    if isinstance(error, ConnectionError):
        return InvokeConnectionError(f"Connection error occurred: {error_message}", 
                                     error_code="CONNECTION_ERROR", http_status=http_status)
    elif isinstance(error, TimeoutError):
        return InvokeTimeoutError(f"Request timed out: {error_message}", 
                                  error_code="TIMEOUT", http_status=http_status)
    else:
        return InvokeAPIError(f"API error occurred: {error_message}", 
                              error_code="UNKNOWN_ERROR", http_status=http_status)