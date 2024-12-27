from typing import Optional, Dict, Any, Type

class InvokeError(Exception):
    """Base class for all invoke errors."""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 http_status: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize an InvokeError.

        Args:
            message (str): The error message.
            error_code (Optional[str]): A code identifying the error type.
            http_status (Optional[int]): The HTTP status code associated with the error.
            details (Optional[Dict[str, Any]]): Additional details about the error.
        """
        self.message: str = message
        self.error_code: Optional[str] = error_code
        self.http_status: Optional[int] = http_status
        self.details: Dict[str, Any] = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Return a string representation of the error.

        Returns:
            str: A formatted string containing error details.
        """
        error_parts: list[str] = [f"Error {self.error_code}: {self.message}" if self.error_code else self.message]
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

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs: Any):
        """
        Initialize an InvokeRateLimitError.

        Args:
            message (str): The error message.
            retry_after (Optional[int]): The number of seconds to wait before retrying.
            **kwargs: Additional keyword arguments to pass to the parent constructor.
        """
        super().__init__(message, **kwargs)
        self.retry_after: Optional[int] = retry_after

    def __str__(self) -> str:
        """
        Return a string representation of the rate limit error.

        Returns:
            str: A formatted string containing error details and retry information.
        """
        base_str: str = super().__str__()
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

    Args:
        error (Exception): The original exception raised by the API.

    Returns:
        InvokeError: An instance of a custom InvokeError subclass.
    """
    error_message: str = str(error)
    http_status: Optional[int] = getattr(error, 'status_code', None) if hasattr(error, 'status_code') else None

    if isinstance(error, ConnectionError):
        return InvokeConnectionError(f"Connection error occurred: {error_message}",
                                     error_code="CONNECTION_ERROR", http_status=http_status)
    elif isinstance(error, TimeoutError):
        return InvokeTimeoutError(f"Request timed out: {error_message}",
                                  error_code="TIMEOUT", http_status=http_status)
    else:
        return InvokeAPIError(f"API error occurred: {error_message}",
                              error_code="UNKNOWN_ERROR", http_status=http_status)

# Type alias for InvokeError and its subclasses
InvokeErrorType = Type[InvokeError]