"""Exceptions for FAIM SDK.

Provides a hierarchy of exceptions for precise error handling and debugging.
"""

from typing import Any

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

    Captures HTTP status codes and server error responses.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: ErrorResponse | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code from response
            response: Parsed error response from backend
            details: Additional context
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.response = response

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.response:
            parts.append(f"response={self.response}")
        if self.details:
            parts.append(f"details={self.details}")
        return " | ".join(parts)


class ModelNotFoundError(APIError):
    """Raised when specified model or version doesn't exist (404).

    Check that:
    - Model name is valid (e.g., 'flowstate', 'toto')
    - Model version is deployed on backend
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
    - Missing required parameters (e.g., horizon, x)
    - Invalid parameter values
    - Incompatible array shapes
    - Model-specific parameter errors
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
    """Raised when authentication fails (401).

    Check that:
    - Token is valid and not expired
    - Token has required permissions
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
