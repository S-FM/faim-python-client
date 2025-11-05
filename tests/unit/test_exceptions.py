"""Unit tests for faim_sdk.exceptions module.

Tests exception hierarchy, error handling, and error contract integration.
"""

import pytest

from faim_client.models.error_code import ErrorCode
from faim_client.models.error_response import ErrorResponse
from faim_sdk.exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    FAIMError,
    InsufficientFundsError,
    InternalServerError,
    ModelNotFoundError,
    NetworkError,
    PayloadTooLargeError,
    RateLimitError,
    SerializationError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)


class TestFAIMError:
    """Tests for base FAIMError exception."""

    def test_initialization_with_message(self):
        """Test basic initialization with message."""
        error = FAIMError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_initialization_with_details(self):
        """Test initialization with details."""
        details = {"key": "value", "count": 42}
        error = FAIMError("Error occurred", details=details)

        assert error.message == "Error occurred"
        assert error.details == details
        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_str_without_details(self):
        """Test string representation without details."""
        error = FAIMError("Test error")
        assert str(error) == "Test error"

    def test_str_with_details(self):
        """Test string representation with details."""
        error = FAIMError("Test error", details={"foo": "bar"})
        result = str(error)

        assert "Test error" in result
        assert "details:" in result
        assert "foo" in result

    def test_inherits_from_exception(self):
        """Test that FAIMError inherits from Exception."""
        error = FAIMError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(FAIMError) as exc_info:
            raise FAIMError("Test error")

        assert exc_info.value.message == "Test error"


class TestSerializationError:
    """Tests for SerializationError exception."""

    def test_inherits_from_faim_error(self):
        """Test inheritance from FAIMError."""
        error = SerializationError("Serialization failed")
        assert isinstance(error, FAIMError)
        assert isinstance(error, Exception)

    def test_initialization(self):
        """Test basic initialization."""
        error = SerializationError("Arrow serialization failed")
        assert error.message == "Arrow serialization failed"

    def test_with_details(self):
        """Test with additional details."""
        error = SerializationError(
            "Invalid array type",
            details={"expected": "ndarray", "got": "list"},
        )
        assert error.details["expected"] == "ndarray"


class TestAPIError:
    """Tests for base APIError exception."""

    def test_inherits_from_faim_error(self):
        """Test inheritance from FAIMError."""
        error = APIError("API failed")
        assert isinstance(error, FAIMError)

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        error = APIError("Request failed")

        assert error.message == "Request failed"
        assert error.status_code is None
        assert error.error_response is None
        assert error.details == {}

    def test_initialization_with_status_code(self):
        """Test initialization with status code."""
        error = APIError("Request failed", status_code=500)
        assert error.status_code == 500

    def test_initialization_with_error_response(self):
        """Test initialization with ErrorResponse."""
        err_response = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Validation failed",
            detail="Invalid parameter",
            request_id="req_123",
        )
        error = APIError("Request failed", error_response=err_response)

        assert error.error_response == err_response
        assert error.error_response.error_code == ErrorCode.VALIDATION_ERROR

    def test_error_code_property_with_response(self):
        """Test error_code property when error_response is available."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Shape error",
        )
        error = APIError("Failed", error_response=err_response)

        assert error.error_code == ErrorCode.INVALID_SHAPE

    def test_error_code_property_without_response(self):
        """Test error_code property when error_response is None."""
        error = APIError("Failed")
        assert error.error_code is None

    def test_str_minimal(self):
        """Test string representation with minimal info."""
        error = APIError("Request failed")
        assert str(error) == "Request failed"

    def test_str_with_status_code(self):
        """Test string representation with status code."""
        error = APIError("Request failed", status_code=422)
        result = str(error)

        assert "Request failed" in result
        assert "status=422" in result

    def test_str_with_error_response(self):
        """Test string representation with error response."""
        err_response = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Validation failed",
            request_id="req_abc123",
        )
        error = APIError("Request failed", status_code=422, error_response=err_response)
        result = str(error)

        assert "Request failed" in result
        assert "status=422" in result
        assert "error_code=validation_error" in result
        assert "request_id=req_abc123" in result

    def test_str_with_all_fields(self):
        """Test string representation with all fields."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Shape error",
            detail="Expected (32, 100, 1)",
            request_id="req_xyz789",
        )
        error = APIError(
            "Request failed",
            status_code=422,
            error_response=err_response,
            details={"retry": True},
        )
        result = str(error)

        assert "Request failed" in result
        assert "status=422" in result
        assert "error_code=invalid_shape" in result
        assert "request_id=req_xyz789" in result
        assert "details=" in result


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = ValidationError("Validation failed")
        assert isinstance(error, APIError)
        assert isinstance(error, FAIMError)

    def test_with_error_code(self):
        """Test with specific error code."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Shape mismatch",
            detail="Expected (batch, seq, feat)",
        )
        error = ValidationError(
            "Validation failed",
            status_code=422,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.INVALID_SHAPE
        assert error.status_code == 422


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = AuthenticationError("Auth failed")
        assert isinstance(error, APIError)

    def test_typical_usage(self):
        """Test typical authentication error scenario."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INVALID_API_KEY,
            message="API key is invalid",
            request_id="req_auth_001",
        )
        error = AuthenticationError(
            "Authentication failed",
            status_code=401,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.INVALID_API_KEY
        assert error.status_code == 401


class TestInsufficientFundsError:
    """Tests for InsufficientFundsError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = InsufficientFundsError("Insufficient funds")
        assert isinstance(error, APIError)

    def test_typical_usage(self):
        """Test typical billing error scenario."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INSUFFICIENT_FUNDS,
            message="Account balance too low",
            detail="Required: $10, Available: $5",
        )
        error = InsufficientFundsError(
            "Billing error",
            status_code=402,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.INSUFFICIENT_FUNDS
        assert error.status_code == 402


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = RateLimitError("Rate limit exceeded")
        assert isinstance(error, APIError)

    def test_typical_usage(self):
        """Test typical rate limit scenario."""
        err_response = ErrorResponse(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message="Too many requests",
            detail="Limit: 100/min",
        )
        error = RateLimitError(
            "Rate limit exceeded",
            status_code=429,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.RATE_LIMIT_EXCEEDED
        assert error.status_code == 429


class TestModelNotFoundError:
    """Tests for ModelNotFoundError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = ModelNotFoundError("Model not found")
        assert isinstance(error, APIError)

    def test_typical_usage(self):
        """Test typical model not found scenario."""
        err_response = ErrorResponse(
            error_code=ErrorCode.MODEL_NOT_FOUND,
            message="Model version not available",
            detail="chronos2:2.0 not found",
        )
        error = ModelNotFoundError(
            "Model not found",
            status_code=404,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.MODEL_NOT_FOUND
        assert error.status_code == 404


class TestPayloadTooLargeError:
    """Tests for PayloadTooLargeError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = PayloadTooLargeError("Payload too large")
        assert isinstance(error, APIError)

    def test_typical_usage(self):
        """Test typical payload size error scenario."""
        err_response = ErrorResponse(
            error_code=ErrorCode.PAYLOAD_TOO_LARGE,
            message="Request size exceeds limit",
            detail="Size: 150MB, Limit: 100MB",
        )
        error = PayloadTooLargeError(
            "Payload too large",
            status_code=413,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.PAYLOAD_TOO_LARGE
        assert error.status_code == 413


class TestInternalServerError:
    """Tests for InternalServerError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = InternalServerError("Internal error")
        assert isinstance(error, APIError)

    def test_typical_usage(self):
        """Test typical internal server error scenario."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred",
            request_id="req_500_abc",
        )
        error = InternalServerError(
            "Internal server error",
            status_code=500,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.INTERNAL_ERROR
        assert error.status_code == 500


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError exception."""

    def test_inherits_from_api_error(self):
        """Test inheritance hierarchy."""
        error = ServiceUnavailableError("Service unavailable")
        assert isinstance(error, APIError)

    def test_typical_usage_503(self):
        """Test typical service unavailable scenario (503)."""
        err_response = ErrorResponse(
            error_code=ErrorCode.TRITON_CONNECTION_ERROR,
            message="Cannot connect to inference backend",
        )
        error = ServiceUnavailableError(
            "Service unavailable",
            status_code=503,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.TRITON_CONNECTION_ERROR
        assert error.status_code == 503

    def test_typical_usage_504(self):
        """Test typical timeout scenario (504)."""
        err_response = ErrorResponse(
            error_code=ErrorCode.TIMEOUT_ERROR,
            message="Request timeout",
        )
        error = ServiceUnavailableError(
            "Gateway timeout",
            status_code=504,
            error_response=err_response,
        )

        assert error.error_code == ErrorCode.TIMEOUT_ERROR
        assert error.status_code == 504


class TestNetworkError:
    """Tests for NetworkError exception."""

    def test_inherits_from_faim_error(self):
        """Test inheritance hierarchy."""
        error = NetworkError("Network failed")
        assert isinstance(error, FAIMError)
        assert not isinstance(error, APIError)

    def test_initialization(self):
        """Test basic initialization."""
        error = NetworkError("Connection refused")
        assert error.message == "Connection refused"

    def test_with_details(self):
        """Test with connection details."""
        error = NetworkError(
            "Connection failed",
            details={"host": "api.example.com", "port": 443},
        )
        assert error.details["host"] == "https://api.faim.it.com"


class TestTimeoutError:
    """Tests for TimeoutError exception."""

    def test_inherits_from_faim_error(self):
        """Test inheritance hierarchy."""
        error = TimeoutError("Request timeout")
        assert isinstance(error, FAIMError)
        assert not isinstance(error, APIError)

    def test_initialization(self):
        """Test basic initialization."""
        error = TimeoutError("Request exceeded 120s timeout")
        assert error.message == "Request exceeded 120s timeout"

    def test_with_details(self):
        """Test with timeout details."""
        error = TimeoutError(
            "Timeout",
            details={"configured": 120, "elapsed": 125},
        )
        assert error.details["configured"] == 120
        assert error.details["elapsed"] == 125


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_inherits_from_faim_error(self):
        """Test inheritance hierarchy."""
        error = ConfigurationError("Config invalid")
        assert isinstance(error, FAIMError)
        assert not isinstance(error, APIError)

    def test_initialization(self):
        """Test basic initialization."""
        error = ConfigurationError("Missing base_url")
        assert error.message == "Missing base_url"

    def test_with_details(self):
        """Test with configuration details."""
        error = ConfigurationError(
            "Invalid configuration",
            details={"field": "base_url", "value": None},
        )
        assert error.details["field"] == "base_url"


class TestExceptionHierarchy:
    """Tests for exception hierarchy and inheritance."""

    def test_all_exceptions_inherit_from_faim_error(self):
        """Test that all SDK exceptions inherit from FAIMError."""
        exceptions = [
            SerializationError("test"),
            APIError("test"),
            ValidationError("test"),
            AuthenticationError("test"),
            InsufficientFundsError("test"),
            RateLimitError("test"),
            ModelNotFoundError("test"),
            PayloadTooLargeError("test"),
            InternalServerError("test"),
            ServiceUnavailableError("test"),
            NetworkError("test"),
            TimeoutError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, FAIMError)
            assert isinstance(exc, Exception)

    def test_api_errors_inherit_from_api_error(self):
        """Test that API exceptions inherit from APIError."""
        api_exceptions = [
            ValidationError("test"),
            AuthenticationError("test"),
            InsufficientFundsError("test"),
            RateLimitError("test"),
            ModelNotFoundError("test"),
            PayloadTooLargeError("test"),
            InternalServerError("test"),
            ServiceUnavailableError("test"),
        ]

        for exc in api_exceptions:
            assert isinstance(exc, APIError)

    def test_non_api_errors_dont_inherit_from_api_error(self):
        """Test that non-API exceptions don't inherit from APIError."""
        non_api_exceptions = [
            SerializationError("test"),
            NetworkError("test"),
            TimeoutError("test"),
            ConfigurationError("test"),
        ]

        for exc in non_api_exceptions:
            assert not isinstance(exc, APIError)

    def test_catch_all_with_faim_error(self):
        """Test that FAIMError can catch all SDK exceptions."""
        test_exceptions = [
            ValidationError("test"),
            NetworkError("test"),
            SerializationError("test"),
        ]

        for exc_class in test_exceptions:
            with pytest.raises(FAIMError):
                raise exc_class

    def test_catch_api_errors_specifically(self):
        """Test catching only API errors."""
        with pytest.raises(APIError):
            raise ValidationError("test")

        # Non-API errors should not be caught
        with pytest.raises(FAIMError):
            try:
                raise NetworkError("test")
            except APIError:
                pytest.fail("NetworkError should not be caught as APIError")


class TestErrorContract:
    """Tests for error contract integration with ErrorResponse and ErrorCode."""

    def test_error_response_integration(self):
        """Test integration with ErrorResponse model."""
        err_response = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Validation failed",
            detail="horizon must be positive",
            request_id="req_123",
            metadata={"field": "horizon", "value": -5},
        )

        error = ValidationError("Request validation failed", error_response=err_response)

        assert error.error_response.error_code == ErrorCode.VALIDATION_ERROR
        assert error.error_response.message == "Validation failed"
        assert error.error_response.detail == "horizon must be positive"
        assert error.error_response.request_id == "req_123"
        assert error.error_response.metadata["field"] == "horizon"

    def test_error_code_enum_access(self):
        """Test accessing ErrorCode enum through exception."""
        err_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Shape error",
        )
        error = ValidationError("Validation failed", error_response=err_response)

        # Can access via property
        assert error.error_code == ErrorCode.INVALID_SHAPE

        # Can compare with enum values
        assert error.error_code == ErrorCode.INVALID_SHAPE
        assert error.error_code != ErrorCode.MISSING_REQUIRED_FIELD

    def test_programmatic_error_handling(self):
        """Test programmatic error handling with error codes."""

        def handle_error(exc: APIError) -> str:
            """Example error handler using error codes."""
            if exc.error_code == ErrorCode.INVALID_SHAPE:
                return "shape_error"
            elif exc.error_code == ErrorCode.MISSING_REQUIRED_FIELD:
                return "missing_field"
            elif exc.error_code == ErrorCode.RATE_LIMIT_EXCEEDED:
                return "rate_limit"
            else:
                return "unknown"

        # Test different error codes
        err1 = ValidationError(
            "Failed",
            error_response=ErrorResponse(error_code=ErrorCode.INVALID_SHAPE, message="Shape error"),
        )
        assert handle_error(err1) == "shape_error"

        err2 = ValidationError(
            "Failed",
            error_response=ErrorResponse(error_code=ErrorCode.MISSING_REQUIRED_FIELD, message="Missing field"),
        )
        assert handle_error(err2) == "missing_field"

        err3 = RateLimitError(
            "Failed",
            error_response=ErrorResponse(error_code=ErrorCode.RATE_LIMIT_EXCEEDED, message="Rate limit"),
        )
        assert handle_error(err3) == "rate_limit"
