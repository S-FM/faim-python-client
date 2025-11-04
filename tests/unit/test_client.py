"""Unit tests for faim_sdk.client module.

Tests ForecastClient initialization, sync/async methods, and error handling.
"""

from unittest.mock import MagicMock, Mock, patch

import httpx
import numpy as np
import pytest

from faim_client.models import ModelName
from faim_client.models.error_code import ErrorCode
from faim_client.models.error_response import ErrorResponse
from faim_sdk.client import ForecastClient
from faim_sdk.exceptions import (
    AuthenticationError,
    ConfigurationError,
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
from faim_sdk.models import Chronos2ForecastRequest, FlowStateForecastRequest, ForecastResponse, TiRexForecastRequest
from faim_sdk.utils import serialize_to_arrow


class TestForecastClientInitialization:
    """Tests for ForecastClient initialization."""

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        client = ForecastClient(base_url="https://api.example.com")

        assert client.base_url == "https://api.example.com"
        assert client.timeout == 120.0
        assert client._client is not None

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        client = ForecastClient(
            base_url="https://api.example.com",
            api_key="test-key-123",
        )

        assert client.base_url == "https://api.example.com"

    def test_initialization_with_timeout(self):
        """Test initialization with custom timeout."""
        client = ForecastClient(
            base_url="https://api.example.com",
            timeout=60.0,
        )

        assert client.timeout == 60.0

    def test_initialization_validation_requires_base_url(self):
        """Test that base_url is required."""
        with pytest.raises(ConfigurationError, match="base_url is required"):
            ForecastClient(base_url="")

    def test_initialization_validation_requires_valid_url(self):
        """Test that base_url must be a valid URL."""
        with pytest.raises(ConfigurationError, match="base_url must be a valid URL"):
            ForecastClient(base_url="not-a-url")

    def test_initialization_validation_positive_timeout(self):
        """Test that timeout must be positive."""
        with pytest.raises(ConfigurationError, match="timeout must be positive"):
            ForecastClient(base_url="https://api.example.com", timeout=0)

    def test_initialization_validation_negative_timeout(self):
        """Test that timeout cannot be negative."""
        with pytest.raises(ConfigurationError, match="timeout must be positive"):
            ForecastClient(base_url="https://api.example.com", timeout=-5)

    def test_context_manager_sync(self):
        """Test synchronous context manager."""
        with ForecastClient(base_url="https://api.example.com") as client:
            assert client._client is not None

        # Client should be closed after context
        # (We can't easily test this without making actual requests)

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test asynchronous context manager."""
        async with ForecastClient(base_url="https://api.example.com") as client:
            assert client._client is not None


class TestForecastClientForecast:
    """Tests for ForecastClient.forecast() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = ForecastClient(base_url="https://api.example.com")
        self.test_data = np.random.rand(2, 10, 1).astype(np.float32)

    @patch("httpx.Client.post")
    def test_forecast_chronos2_point(self, mock_post):
        """Test forecast with Chronos2 model for point predictions."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        response_metadata = {"model_name": "chronos2", "model_version": "1"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, response_metadata)
        mock_post.return_value = mock_response

        # Create request and call forecast
        request = Chronos2ForecastRequest(
            x=self.test_data,
            horizon=5,
            output_type="point",
        )
        response = self.client.forecast(request)

        # Verify response
        assert isinstance(response, ForecastResponse)
        assert response.point is not None
        assert response.point.shape == (2, 5, 1)
        assert response.metadata["model_name"] == "chronos2"

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "chronos2" in call_args[1]["url"]

    @patch("httpx.Client.post")
    def test_forecast_chronos2_quantiles(self, mock_post):
        """Test forecast with Chronos2 model for quantile predictions."""
        # Setup mock response
        response_arrays = {"quantiles": np.random.rand(2, 5, 3).astype(np.float32)}
        response_metadata = {"model_name": "chronos2", "quantiles": [0.1, 0.5, 0.9]}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, response_metadata)
        mock_post.return_value = mock_response

        # Create request and call forecast
        request = Chronos2ForecastRequest(
            x=self.test_data,
            horizon=5,
            output_type="quantiles",
            quantiles=[0.1, 0.5, 0.9],
        )
        response = self.client.forecast(request)

        # Verify response
        assert response.quantiles is not None
        assert response.quantiles.shape == (2, 5, 3)
        assert response.point is None

    @patch("httpx.Client.post")
    def test_forecast_flowstate(self, mock_post):
        """Test forecast with FlowState model."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        response_metadata = {"model_name": "flowstate"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, response_metadata)
        mock_post.return_value = mock_response

        # Create request and call forecast
        request = FlowStateForecastRequest(
            x=self.test_data,
            horizon=5,
            output_type="point",
            prediction_type="mean",
        )
        response = self.client.forecast(request)

        # Verify response
        assert response.point is not None
        assert response.metadata["model_name"] == "flowstate"

        # Verify URL contains flowstate
        call_args = mock_post.call_args
        assert "flowstate" in call_args[1]["url"]

    @patch("httpx.Client.post")
    def test_forecast_tirex(self, mock_post):
        """Test forecast with TiRex model."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        response_metadata = {"model_name": "tirex"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, response_metadata)
        mock_post.return_value = mock_response

        # Create request and call forecast
        request = TiRexForecastRequest(
            x=self.test_data,
            horizon=5,
            output_type="point",
        )
        response = self.client.forecast(request)

        # Verify response
        assert response.point is not None

        # Verify URL contains tirex
        call_args = mock_post.call_args
        assert "tirex" in call_args[1]["url"]

    @patch("httpx.Client.post")
    def test_forecast_with_custom_model_version(self, mock_post):
        """Test forecast with custom model version."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, {})
        mock_post.return_value = mock_response

        # Create request with custom version
        request = Chronos2ForecastRequest(
            x=self.test_data,
            horizon=5,
            model_version="2.0",
        )
        self.client.forecast(request)

        # Verify URL contains version
        call_args = mock_post.call_args
        assert "/2.0" in call_args[1]["url"] or "2.0" in call_args[1]["url"]


class TestForecastClientErrorHandling:
    """Tests for ForecastClient error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = ForecastClient(base_url="https://api.example.com")
        self.test_data = np.random.rand(2, 10, 1).astype(np.float32)
        self.request = Chronos2ForecastRequest(x=self.test_data, horizon=5)

    @patch("httpx.Client.post")
    def test_validation_error_422(self, mock_post):
        """Test handling of 422 validation errors."""
        # Setup mock error response
        error_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Shape validation failed",
            detail="Expected (batch, seq, feat)",
            request_id="req_123",
        )
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(ValidationError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 422
        assert exc_info.value.error_code == ErrorCode.INVALID_SHAPE

    @patch("httpx.Client.post")
    def test_authentication_error_401(self, mock_post):
        """Test handling of 401 authentication errors."""
        error_response = ErrorResponse(
            error_code=ErrorCode.INVALID_API_KEY,
            message="API key is invalid",
        )
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(AuthenticationError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 401

    @patch("httpx.Client.post")
    def test_authentication_error_403(self, mock_post):
        """Test handling of 403 forbidden errors."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        with pytest.raises(AuthenticationError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 403

    @patch("httpx.Client.post")
    def test_insufficient_funds_error_402(self, mock_post):
        """Test handling of 402 payment required errors."""
        error_response = ErrorResponse(
            error_code=ErrorCode.INSUFFICIENT_FUNDS,
            message="Insufficient balance",
        )
        mock_response = Mock()
        mock_response.status_code = 402
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(InsufficientFundsError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 402

    @patch("httpx.Client.post")
    def test_model_not_found_error_404(self, mock_post):
        """Test handling of 404 not found errors."""
        error_response = ErrorResponse(
            error_code=ErrorCode.MODEL_NOT_FOUND,
            message="Model not found",
        )
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(ModelNotFoundError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 404

    @patch("httpx.Client.post")
    def test_payload_too_large_error_413(self, mock_post):
        """Test handling of 413 payload too large errors."""
        error_response = ErrorResponse(
            error_code=ErrorCode.PAYLOAD_TOO_LARGE,
            message="Payload exceeds limit",
        )
        mock_response = Mock()
        mock_response.status_code = 413
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(PayloadTooLargeError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 413

    @patch("httpx.Client.post")
    def test_rate_limit_error_429(self, mock_post):
        """Test handling of 429 rate limit errors."""
        error_response = ErrorResponse(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message="Rate limit exceeded",
        )
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(RateLimitError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 429

    @patch("httpx.Client.post")
    def test_internal_server_error_500(self, mock_post):
        """Test handling of 500 internal server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        with pytest.raises(InternalServerError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 500

    @patch("httpx.Client.post")
    def test_service_unavailable_error_503(self, mock_post):
        """Test handling of 503 service unavailable errors."""
        error_response = ErrorResponse(
            error_code=ErrorCode.TRITON_CONNECTION_ERROR,
            message="Cannot connect to backend",
        )
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        with pytest.raises(ServiceUnavailableError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 503

    @patch("httpx.Client.post")
    def test_service_unavailable_error_504(self, mock_post):
        """Test handling of 504 gateway timeout errors."""
        mock_response = Mock()
        mock_response.status_code = 504
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        with pytest.raises(ServiceUnavailableError) as exc_info:
            self.client.forecast(self.request)

        assert exc_info.value.status_code == 504

    @patch("httpx.Client.post")
    def test_network_error_connection_failed(self, mock_post):
        """Test handling of network connection errors."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(NetworkError) as exc_info:
            self.client.forecast(self.request)

        assert "Connection refused" in str(exc_info.value)

    @patch("httpx.Client.post")
    def test_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        mock_post.side_effect = httpx.TimeoutException("Request timeout")

        with pytest.raises(TimeoutError) as exc_info:
            self.client.forecast(self.request)

        assert "timeout" in str(exc_info.value).lower()

    @patch("faim_sdk.utils.serialize_to_arrow")
    def test_serialization_error(self, mock_serialize):
        """Test handling of serialization errors."""
        mock_serialize.side_effect = TypeError("Invalid array type")

        with pytest.raises(SerializationError) as exc_info:
            self.client.forecast(self.request)

        assert "serialization" in str(exc_info.value).lower()


class TestForecastClientAsync:
    """Tests for ForecastClient.forecast_async() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = ForecastClient(base_url="https://api.example.com")
        self.test_data = np.random.rand(2, 10, 1).astype(np.float32)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_forecast_async_success(self, mock_post):
        """Test async forecast with successful response."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        response_metadata = {"model_name": "chronos2"}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, response_metadata)
        mock_post.return_value = mock_response

        # Create request and call async forecast
        request = Chronos2ForecastRequest(
            x=self.test_data,
            horizon=5,
            output_type="point",
        )
        response = await self.client.forecast_async(request)

        # Verify response
        assert isinstance(response, ForecastResponse)
        assert response.point is not None
        assert response.point.shape == (2, 5, 1)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_forecast_async_validation_error(self, mock_post):
        """Test async forecast with validation error."""
        # Setup mock error response
        error_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Shape validation failed",
        )
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = error_response.to_dict()
        mock_post.return_value = mock_response

        request = Chronos2ForecastRequest(x=self.test_data, horizon=5)

        with pytest.raises(ValidationError):
            await self.client.forecast_async(request)


class TestForecastClientClose:
    """Tests for ForecastClient resource cleanup."""

    def test_close_method(self):
        """Test close method closes underlying client."""
        client = ForecastClient(base_url="https://api.example.com")

        # Should not raise any exceptions
        client.close()

    @pytest.mark.asyncio
    async def test_aclose_method(self):
        """Test async close method."""
        client = ForecastClient(base_url="https://api.example.com")

        # Should not raise any exceptions
        await client.aclose()

    def test_context_manager_closes_client(self):
        """Test that context manager closes client on exit."""
        with ForecastClient(base_url="https://api.example.com") as client:
            pass

        # Client should be closed after context exits
        # (We can't easily verify this without implementation details)

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self):
        """Test that async context manager closes client on exit."""
        async with ForecastClient(base_url="https://api.example.com") as client:
            pass

        # Client should be closed after context exits


class TestForecastClientLogging:
    """Tests for ForecastClient logging behavior."""

    @patch("httpx.Client.post")
    def test_logs_request_info(self, mock_post):
        """Test that client logs request information."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, {})
        mock_post.return_value = mock_response

        client = ForecastClient(base_url="https://api.example.com")
        request = Chronos2ForecastRequest(
            x=np.random.rand(2, 10, 1).astype(np.float32),
            horizon=5,
        )

        # Should not raise exceptions
        with patch("faim_sdk.client.logger") as mock_logger:
            client.forecast(request)
            # Verify some logging occurred
            assert mock_logger.debug.called or mock_logger.info.called

    @patch("httpx.Client.post")
    def test_logs_error_info(self, mock_post):
        """Test that client logs error information."""
        # Setup mock error response
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        client = ForecastClient(base_url="https://api.example.com")
        request = Chronos2ForecastRequest(
            x=np.random.rand(2, 10, 1).astype(np.float32),
            horizon=5,
        )

        with patch("faim_sdk.client.logger") as mock_logger:
            try:
                client.forecast(request)
            except ValidationError:
                pass

            # Verify error logging occurred
            assert mock_logger.error.called or mock_logger.warning.called


class TestForecastClientIntegration:
    """Integration-style tests for realistic usage patterns."""

    @patch("httpx.Client.post")
    def test_multiple_requests_same_client(self, mock_post):
        """Test making multiple requests with the same client instance."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, {})
        mock_post.return_value = mock_response

        client = ForecastClient(base_url="https://api.example.com")
        data = np.random.rand(2, 10, 1).astype(np.float32)

        # Make multiple requests
        for _ in range(3):
            request = Chronos2ForecastRequest(x=data, horizon=5)
            response = client.forecast(request)
            assert response.point is not None

        # Verify all requests were made
        assert mock_post.call_count == 3

    @patch("httpx.Client.post")
    def test_different_models_same_client(self, mock_post):
        """Test using different models with the same client."""
        # Setup mock response
        response_arrays = {"point": np.random.rand(2, 5, 1).astype(np.float32)}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = serialize_to_arrow(response_arrays, {})
        mock_post.return_value = mock_response

        client = ForecastClient(base_url="https://api.example.com")
        data = np.random.rand(2, 10, 1).astype(np.float32)

        # Test different models
        requests = [
            Chronos2ForecastRequest(x=data, horizon=5),
            FlowStateForecastRequest(x=data, horizon=5, prediction_type="mean"),
            TiRexForecastRequest(x=data, horizon=5),
        ]

        for req in requests:
            response = client.forecast(req)
            assert response.point is not None

        assert mock_post.call_count == 3
