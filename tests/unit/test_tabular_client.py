"""Unit tests for faim_sdk.tabular_client module.

Tests TabularClient initialization, sync/async methods, and error handling.
"""

from unittest.mock import Mock, patch

import httpx
import numpy as np
import pytest

from faim_client.models.error_code import ErrorCode
from faim_client.models.error_response import ErrorResponse
from faim_sdk.exceptions import (
    APIError,
    AuthenticationError,
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
from faim_sdk.models import LimiXPredictRequest
from faim_sdk.tabular_client import TabularClient


class TestTabularClientInitialization:
    """Tests for TabularClient initialization."""

    def test_initialization_minimal(self):
        """Test initialization with minimal parameters."""
        client = TabularClient()

        assert client.base_url == "https://api.faim.it.com"
        assert client._client is not None

    def test_initialization_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        custom_url = "https://api.example.com"
        client = TabularClient(base_url=custom_url)

        assert client.base_url == custom_url

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        client = TabularClient(
            api_key="test-key-123",
        )

        assert client.base_url == "https://api.faim.it.com"
        # API key is stored in the authenticated client
        assert client._client is not None

    def test_initialization_with_timeout(self):
        """Test initialization with custom timeout."""
        client = TabularClient(
            base_url="https://api.example.com",
            timeout=30.0,
        )

        # Timeout is stored as httpx.Timeout object
        assert client._client._timeout is not None

    def test_initialization_with_ssl_verification_disabled(self):
        """Test initialization with SSL verification disabled."""
        client = TabularClient(
            base_url="https://api.example.com",
            verify_ssl=False,
        )

        assert client._client is not None

    def test_initialization_with_httpx_kwargs(self):
        """Test initialization with additional httpx kwargs."""
        custom_headers = {"X-Custom-Header": "test-value"}
        client = TabularClient(
            base_url="https://api.example.com",
            headers=custom_headers,
        )

        assert client._client is not None

    def test_context_manager_sync(self):
        """Test synchronous context manager."""
        with TabularClient(base_url="https://api.example.com") as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test asynchronous context manager."""
        async with TabularClient(base_url="https://api.example.com") as client:
            assert client._client is not None


class TestTabularClientPredict:
    """Tests for TabularClient.predict() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TabularClient(base_url="https://api.example.com")
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 2, 100).astype(np.float32)
        self.X_test = np.random.randn(20, 10).astype(np.float32)

    def teardown_method(self):
        """Clean up after tests."""
        self.client.close()

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_predict_classification_with_mock(self, mock_api):
        """Test successful classification prediction."""
        # Create mock response
        predictions = np.array([0, 1, 0, 1], dtype=np.float32)
        probabilities = np.random.rand(4, 2).astype(np.float32)

        # Mock the API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"mock_arrow_data"

        with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
            mock_deser.return_value = (
                {"predictions": predictions, "probabilities": probabilities},
                {"model_name": "LimiX", "model_version": "1.0", "task_type": "Classification"},
            )

            mock_api.return_value = mock_response

            request = LimiXPredictRequest(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                task_type="Classification",
            )

            response = self.client.predict(request)

            assert response is not None
            mock_api.assert_called_once()

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_predict_regression_with_mock(self, mock_api):
        """Test successful regression prediction."""
        predictions = np.random.rand(20).astype(np.float32)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"mock_arrow_data"

        with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
            mock_deser.return_value = (
                {"predictions": predictions},
                {"model_name": "LimiX", "model_version": "1.0", "task_type": "Regression"},
            )

            mock_api.return_value = mock_response

            request = LimiXPredictRequest(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                task_type="Regression",
            )

            response = self.client.predict(request)

            assert response is not None
            mock_api.assert_called_once()

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_predict_calls_serialization(self, mock_api):
        """Test that predict properly serializes the request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"mock_arrow_data"

        with patch("faim_sdk.tabular_client.serialize_to_arrow_tabular") as mock_ser:
            with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
                mock_ser.return_value = b"serialized_data"
                mock_deser.return_value = (
                    {"predictions": np.array([0, 1])},
                    {"model_name": "LimiX", "model_version": "1.0", "task_type": "Classification"},
                )
                mock_api.return_value = mock_response

                request = LimiXPredictRequest(
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    task_type="Classification",
                )

                self.client.predict(request)

                mock_ser.assert_called_once()


class TestTabularClientPredictAsync:
    """Tests for TabularClient.predict_async() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TabularClient(base_url="https://api.example.com")
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 2, 100).astype(np.float32)
        self.X_test = np.random.randn(20, 10).astype(np.float32)

    def teardown_method(self):
        """Clean up after tests."""
        self.client.close()

    @pytest.mark.asyncio
    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.asyncio_detailed")
    async def test_predict_async_classification(self, mock_api):
        """Test async classification prediction."""
        predictions = np.array([0, 1, 0, 1], dtype=np.float32)
        probabilities = np.random.rand(4, 2).astype(np.float32)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"mock_arrow_data"

        # Make the mock async compatible
        mock_api.return_value = mock_response

        with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
            mock_deser.return_value = (
                {"predictions": predictions, "probabilities": probabilities},
                {"model_name": "LimiX", "model_version": "1.0", "task_type": "Classification"},
            )

            request = LimiXPredictRequest(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                task_type="Classification",
            )

            response = await self.client.predict_async(request)

            assert response is not None
            mock_api.assert_called_once()

    @pytest.mark.asyncio
    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.asyncio_detailed")
    async def test_predict_async_regression(self, mock_api):
        """Test async regression prediction."""
        predictions = np.random.rand(20).astype(np.float32)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"mock_arrow_data"

        mock_api.return_value = mock_response

        with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
            mock_deser.return_value = (
                {"predictions": predictions},
                {"model_name": "LimiX", "model_version": "1.0", "task_type": "Regression"},
            )

            request = LimiXPredictRequest(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                task_type="Regression",
            )

            response = await self.client.predict_async(request)

            assert response is not None


class TestTabularClientErrorHandling:
    """Tests for error handling in TabularClient."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TabularClient(base_url="https://api.example.com")
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 2, 100).astype(np.float32)
        self.X_test = np.random.randn(20, 10).astype(np.float32)

    def teardown_method(self):
        """Clean up after tests."""
        self.client.close()

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_authentication_error_401(self, mock_api):
        """Test handling of 401 Unauthorized error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            message="Invalid API key",
            detail="The provided API key is invalid",
            request_id="req_123",
        )

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_response is not None

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_authentication_error_403(self, mock_api):
        """Test handling of 403 Forbidden error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            message="Access denied",
            detail="You don't have permission to access this resource",
            request_id="req_124",
        )

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 403

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_insufficient_funds_error(self, mock_api):
        """Test handling of 402 Payment Required error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.INSUFFICIENT_FUNDS,
            message="Insufficient balance",
            detail="Your account balance is insufficient for this operation",
            request_id="req_125",
        )

        mock_response = Mock()
        mock_response.status_code = 402
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(InsufficientFundsError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 402

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_model_not_found_error(self, mock_api):
        """Test handling of 404 Not Found error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.MODEL_NOT_FOUND,
            message="Model not found",
            detail="The requested model 'InvalidModel' with version '1.0' does not exist",
            request_id="req_126",
        )

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(ModelNotFoundError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 404

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_payload_too_large_error(self, mock_api):
        """Test handling of 413 Payload Too Large error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.REQUEST_TOO_LARGE,
            message="Request too large",
            detail="The request payload exceeds the maximum allowed size",
            request_id="req_127",
        )

        mock_response = Mock()
        mock_response.status_code = 413
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(PayloadTooLargeError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 413

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_validation_error_422(self, mock_api):
        """Test handling of 422 Unprocessable Entity error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Invalid input shape",
            detail="X_train shape (100, 10) does not match X_test shape (20, 15)",
            request_id="req_128",
        )

        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(ValidationError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 422

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_rate_limit_error(self, mock_api):
        """Test handling of 429 Too Many Requests error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message="Rate limit exceeded",
            detail="Too many requests. Please retry after 60 seconds",
            request_id="req_129",
        )

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(RateLimitError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 429

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_internal_server_error(self, mock_api):
        """Test handling of 500 Internal Server Error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Internal server error",
            detail="An unexpected error occurred on the server",
            request_id="req_130",
        )

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(InternalServerError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 500

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_service_unavailable_error_503(self, mock_api):
        """Test handling of 503 Service Unavailable error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.RESOURCE_EXHAUSTED,
            message="Service unavailable",
            detail="The service is temporarily unavailable. Please try again later",
            request_id="req_131",
        )

        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(ServiceUnavailableError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 503

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_service_unavailable_error_504(self, mock_api):
        """Test handling of 504 Gateway Timeout error."""
        error_response = ErrorResponse(
            error_code=ErrorCode.TIMEOUT_ERROR,
            message="Gateway timeout",
            detail="The gateway timed out waiting for the backend",
            request_id="req_132",
        )

        mock_response = Mock()
        mock_response.status_code = 504
        mock_response.content = b"error response"
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(ServiceUnavailableError) as exc_info:
            self.client.predict(request)

        assert exc_info.value.status_code == 504

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_serialization_error_on_request(self, mock_api):
        """Test handling of serialization error on request."""
        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with patch("faim_sdk.tabular_client.serialize_to_arrow_tabular") as mock_ser:
            mock_ser.side_effect = ValueError("Invalid arrow data")

            with pytest.raises(SerializationError) as exc_info:
                self.client.predict(request)

            assert "Failed to serialize request" in str(exc_info.value)

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_serialization_error_on_response(self, mock_api):
        """Test handling of serialization error on response deserialization."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"invalid_arrow_data"

        mock_api.return_value = mock_response

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
            mock_deser.side_effect = ValueError("Invalid arrow format")

            with pytest.raises(SerializationError) as exc_info:
                self.client.predict(request)

            assert "Failed to deserialize response" in str(exc_info.value)

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_network_error(self, mock_api):
        """Test handling of network errors."""
        mock_api.side_effect = httpx.NetworkError("Connection failed")

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(NetworkError) as exc_info:
            self.client.predict(request)

        assert "Network communication failed" in str(exc_info.value)

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_timeout_error(self, mock_api):
        """Test handling of timeout errors."""
        mock_api.side_effect = httpx.TimeoutException("Request timeout")

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(TimeoutError) as exc_info:
            self.client.predict(request)

        assert "Request exceeded timeout" in str(exc_info.value)

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_unexpected_exception(self, mock_api):
        """Test handling of unexpected exceptions."""
        mock_api.side_effect = RuntimeError("Unexpected error")

        request = LimiXPredictRequest(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            task_type="Classification",
        )

        with pytest.raises(APIError) as exc_info:
            self.client.predict(request)

        assert "Unexpected error" in str(exc_info.value)


class TestTabularClientResourceManagement:
    """Tests for resource cleanup methods."""

    def test_close_method(self):
        """Test close() method."""
        client = TabularClient(base_url="https://api.example.com")
        client.close()  # Should not raise any errors

    @pytest.mark.asyncio
    async def test_aclose_method(self):
        """Test aclose() async method."""
        client = TabularClient(base_url="https://api.example.com")
        await client.aclose()  # Should not raise any errors

    def test_context_manager_cleanup(self):
        """Test context manager cleanup."""
        with TabularClient(base_url="https://api.example.com") as client:
            assert client._client is not None

        # After exiting context, client should be closed
        # (Can't directly test this without making actual requests)

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self):
        """Test async context manager cleanup."""
        async with TabularClient(base_url="https://api.example.com") as client:
            assert client._client is not None

        # After exiting context, client should be closed


class TestTabularClientLogging:
    """Tests for logging behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TabularClient(base_url="https://api.example.com")
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 2, 100).astype(np.float32)
        self.X_test = np.random.randn(20, 10).astype(np.float32)

    def teardown_method(self):
        """Clean up after tests."""
        self.client.close()

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_successful_prediction_logs_info(self, mock_api, caplog):
        """Test that successful predictions are logged at info level."""
        import logging

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"mock_arrow_data"

        with patch("faim_sdk.tabular_client.serialize_to_arrow_tabular") as mock_ser:
            with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
                mock_ser.return_value = b"serialized_data"
                mock_deser.return_value = (
                    {"predictions": np.array([0, 1])},
                    {"model_name": "LimiX", "model_version": "1.0", "task_type": "Classification"},
                )
                mock_api.return_value = mock_response

                with caplog.at_level(logging.DEBUG):
                    request = LimiXPredictRequest(
                        X_train=self.X_train,
                        y_train=self.y_train,
                        X_test=self.X_test,
                        task_type="Classification",
                    )

                    self.client.predict(request)

                # Should have debug logs for prediction start
                log_messages = [record.message for record in caplog.records]
                assert any("Starting tabular prediction" in msg for msg in log_messages)

    @patch("faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed")
    def test_error_prediction_logs_error(self, mock_api, caplog):
        """Test that errors are logged at error level."""
        import logging

        error_response = ErrorResponse(
            error_code=ErrorCode.INVALID_SHAPE,
            message="Invalid shape",
            detail="Wrong dimensions",
            request_id="req_999",
        )

        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.parsed = error_response

        mock_api.return_value = mock_response

        with caplog.at_level(logging.ERROR):
            request = LimiXPredictRequest(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                task_type="Classification",
            )

            with pytest.raises(ValidationError):
                self.client.predict(request)

            # Should have error logs
            log_messages = [record.message for record in caplog.records]
            assert any("API error" in msg for msg in log_messages)


class TestTabularClientIntegration:
    """Integration tests for TabularClient."""

    def test_multiple_predictions_with_same_client(self):
        """Test making multiple predictions with the same client."""
        client = TabularClient(base_url="https://api.example.com")

        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 100).astype(np.float32)
        X_test1 = np.random.randn(20, 10).astype(np.float32)
        X_test2 = np.random.randn(30, 10).astype(np.float32)

        with patch(
            "faim_sdk.tabular_client.predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed"
        ) as mock_api:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"mock_arrow_data"

            with patch("faim_sdk.tabular_client.deserialize_from_arrow_tabular") as mock_deser:
                mock_deser.return_value = (
                    {"predictions": np.array([0, 1, 0, 1])},
                    {"model_name": "LimiX", "model_version": "1.0", "task_type": "Classification"},
                )
                mock_api.return_value = mock_response

                request1 = LimiXPredictRequest(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test1,
                    task_type="Classification",
                )
                response1 = client.predict(request1)

                request2 = LimiXPredictRequest(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test2,
                    task_type="Classification",
                )
                response2 = client.predict(request2)

                assert response1 is not None
                assert response2 is not None
                assert mock_api.call_count == 2

        client.close()

    def test_client_with_api_key_initialization(self):
        """Test client initialization with API key."""
        client = TabularClient(
            base_url="https://api.example.com",
            api_key="test-api-key-123",
        )

        assert client._client is not None
        client.close()
