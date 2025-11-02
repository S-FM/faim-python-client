"""Exceptions for FAIM SDK.

Provides a hierarchy of exceptions for precise error handling and debugging.
Integrates with the unified error contract via generated ErrorResponse and ErrorCode.
"""

from typing import Any

from faim_client.models.error_code import ErrorCode
from faim_client.models.error_response import ErrorResponse


class FAIMError(Exception):
    """Base exception for all FAIM SDK errors.

    All FAIM SDK exceptions inherit from this class, allowing catch-all
    error handling when needed.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize FAIM error.

        Args:
            message: Human-readable error message
            details: Additional context for debugging (logged but not exposed to end users)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error.

        Returns:
            Error message with details if available
        """
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class SerializationError(FAIMError):
    """Raised when Arrow serialization/deserialization fails.

    This typically indicates:
    - Invalid numpy array types
    - Corrupted Arrow stream
    - Incompatible Arrow schema
    """

    pass


class APIError(FAIMError):
    """Base exception for API-related errors.

    Captures HTTP status codes and server error responses with error contract integration.
    All API errors include an optional ErrorResponse with machine-readable error codes.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_response: ErrorResponse | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code from response
            error_response: Parsed error response from backend (includes error_code, message, detail)
            details: Additional context for debugging
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.error_response = error_response

    @property
    def error_code(self) -> ErrorCode | None:
        """Get machine-readable error code from error response.

        Returns:
            ErrorCode enum value if available, None otherwise

        Example:
            try:
                client.forecast(...)
            except ValidationError as e:
                if e.error_code == ErrorCode.INVALID_SHAPE:
                    # Handle shape error specifically
                    pass
        """
        return self.error_response.error_code if self.error_response else None

    def __str__(self) -> str:
        """Return detailed string representation of the API error.

        Returns:
            Formatted string with message, status code, error code, request ID, and details
        """
        parts = [self.message]
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.error_response:
            parts.append(f"error_code={self.error_response.error_code}")
            if self.error_response.request_id:
                parts.append(f"request_id={self.error_response.request_id}")
        if self.details:
            parts.append(f"details={self.details}")
        return " | ".join(parts)


class ModelNotFoundError(APIError):
    """Raised when specified model or version doesn't exist (404).

    Check that:
    - Model name is valid (e.g., 'flowstate', 'chronos2', 'tirex')
    - Model version is deployed on backend

    Error codes:
    - MODEL_NOT_FOUND
    - PRICING_NOT_FOUND
    - RESOURCE_NOT_FOUND
    """

    pass


class PayloadTooLargeError(APIError):
    """Raised when request payload exceeds backend size limit (413).

    Consider:
    - Reducing batch size
    - Reducing sequence length
    - Splitting request into multiple calls
    """

    pass


class ValidationError(APIError):
    """Raised when backend rejects request as invalid (422).

    Common causes:
    - Missing required parameters (e.g., horizon, x, output_type)
    - Invalid parameter values
    - Incompatible array shapes
    - Model-specific parameter errors

    Error codes:
    - VALIDATION_ERROR
    - INVALID_MODEL_INPUT
    - INVALID_PARAMETER
    - MISSING_REQUIRED_FIELD
    - INVALID_SHAPE
    - INVALID_DTYPE
    - INVALID_VALUE_RANGE
    """

    pass


class InternalServerError(APIError):
    """Raised when backend encounters internal error (500).

    This indicates a backend issue. Check:
    - Backend logs for stack traces
    - Model health and resource availability
    """

    pass


class NetworkError(FAIMError):
    """Raised when network communication fails.

    Common causes:
    - Connection timeout
    - DNS resolution failure
    - Network unreachable
    """

    pass


class TimeoutError(FAIMError):
    """Raised when request exceeds configured timeout.

    Consider:
    - Increasing client timeout
    - Reducing batch size
    - Checking backend performance
    """

    pass


class AuthenticationError(APIError):
    """Raised when authentication fails (401, 403).

    Check that:
    - API key is valid and not expired
    - API key has required permissions

    Error codes:
    - AUTHENTICATION_REQUIRED
    - AUTHENTICATION_FAILED
    - INVALID_API_KEY
    - AUTHORIZATION_FAILED
    """

    pass


class InsufficientFundsError(APIError):
    """Raised when billing account balance is insufficient (402).

    This indicates the user's account doesn't have enough credits
    to perform the requested inference.

    Actions:
    - Add credits to billing account
    - Check pricing for the model/version being used

    Error codes:
    - INSUFFICIENT_FUNDS
    - BILLING_TRANSACTION_FAILED
    """

    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded (429).

    The user has made too many requests in a short period.

    Actions:
    - Implement exponential backoff retry logic
    - Reduce request frequency
    - Contact support for rate limit increases

    Error codes:
    - RATE_LIMIT_EXCEEDED
    """

    pass


class ServiceUnavailableError(APIError):
    """Raised when service is temporarily unavailable (503, 504).

    This typically indicates transient infrastructure issues that
    may be resolved by retrying with exponential backoff.

    Common causes:
    - Triton server connection failures
    - GPU/CPU resources exhausted
    - Out of memory conditions

    Error codes:
    - TRITON_CONNECTION_ERROR
    - RESOURCE_EXHAUSTED
    - OUT_OF_MEMORY
    - TIMEOUT_ERROR (504)
    """

    pass


class ConfigurationError(FAIMError):
    """Raised when SDK is misconfigured.

    Common causes:
    - Missing required configuration
    - Invalid parameter combinations
    - Malformed base URL
    """

    pass
