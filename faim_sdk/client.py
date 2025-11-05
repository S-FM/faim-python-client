"""FAIM SDK client for time-series forecasting.

Provides high-level, type-safe API with automatic serialization, error handling,
and observability.
"""

import io
import json
import logging
import warnings
from copy import copy

import httpx
import numpy as np

from faim_client import AuthenticatedClient, Client
from faim_client.api.forecast import forecast_v1_ts_forecast_model_name_model_version_post
from faim_client.models import ModelName
from faim_client.models.error_response import ErrorResponse
from faim_client.types import File

from .exceptions import (
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
from .models import ForecastRequest, ForecastResponse
from .utils import deserialize_from_arrow, serialize_to_arrow

logger = logging.getLogger(__name__)


def _parse_error_response(response) -> ErrorResponse | None:
    """Parse ErrorResponse from HTTP response body.

    Args:
        response: HTTP response object from generated client

    Returns:
        Parsed ErrorResponse if available, None otherwise
    """
    try:
        # Try parsing from response.parsed first (generated client parsing)
        if hasattr(response, "parsed") and isinstance(response.parsed, ErrorResponse):
            return response.parsed

        # Fallback: try parsing JSON content directly
        if hasattr(response, "content") and response.content:
            error_dict = json.loads(response.content)
            return ErrorResponse.from_dict(error_dict)
    except Exception as e:
        logger.warning(f"Failed to parse error response: {e}")

    return None


def _needs_univariate_transformation(request: ForecastRequest) -> bool:
    """Check if request requires univariate transformation.

    FlowState and TiRex models only support univariate forecasting.
    When they receive multivariate input (features > 1), the input
    must be transformed to forecast each feature independently.

    Args:
        request: Forecast request to check

    Returns:
        True if transformation is needed, False otherwise
    """
    # Only FlowState and TiRex require transformation
    if request.model_name not in (ModelName.FLOWSTATE, ModelName.TIREX):
        return False

    # Check if input is multivariate (features > 1)
    num_features = request.x.shape[2]  # Shape is (batch, seq_len, features)
    return num_features > 1


def _prepare_univariate_request(request: ForecastRequest) -> tuple[ForecastRequest, tuple[int, int]]:
    """Prepare request for univariate-only models with multivariate input.

    Transforms input from (batch, seq_len, features) to (batch*features, seq_len, 1)
    and issues a warning to the user that features will be forecast independently.

    Args:
        request: Original forecast request with multivariate input

    Returns:
        Tuple of (modified request, (original_batch_size, num_features))

    Example:
        Input shape:  (batch=2, seq_len=100, features=3)
        Output shape: (batch=6, seq_len=100, features=1)
        Mapping:
            - Feature 0 of series 0 → new batch index 0
            - Feature 1 of series 0 → new batch index 1
            - Feature 2 of series 0 → new batch index 2
            - Feature 0 of series 1 → new batch index 3
            - Feature 1 of series 1 → new batch index 4
            - Feature 2 of series 1 → new batch index 5
    """
    original_batch_size, seq_len, num_features = request.x.shape

    # Issue user warning
    warnings.warn(
        f"{request.model_name.value.title()} model only supports univariate forecasting. "
        f"Input with {num_features} features will be forecast independently. "
        f"Each feature will be treated as a separate time series.",
        UserWarning,
        stacklevel=3,  # Point to the user's code, not this internal function
    )

    logger.info(
        f"Transforming multivariate input for {request.model_name.value}: "
        f"shape {request.x.shape} → ({original_batch_size * num_features}, {seq_len}, 1)"
    )

    # Reshape: (batch, seq_len, features) → (batch, features, seq_len) → (batch*features, seq_len) → (batch*features, seq_len, 1)
    # We want to interleave features across the batch dimension
    x_transposed = request.x.transpose(0, 2, 1)  # (batch, features, seq_len)
    x_flattened = x_transposed.reshape(original_batch_size * num_features, seq_len)  # (batch*features, seq_len)
    x_univariate = x_flattened[:, :, np.newaxis]  # (batch*features, seq_len, 1)

    # Create modified request with reshaped x
    # Use copy to avoid modifying the original request
    modified_request = copy(request)
    modified_request.x = x_univariate

    return modified_request, (original_batch_size, num_features)


def _reshape_univariate_response(
    response: ForecastResponse,
    original_batch_size: int,
    num_features: int,
) -> ForecastResponse:
    """Reshape response from univariate transformation back to multivariate format.

    Reverses the transformation done by _prepare_univariate_request() to restore
    the original batch and feature dimensions.

    Args:
        response: Response from server with univariate format
        original_batch_size: Original batch size before transformation
        num_features: Number of features in original input

    Returns:
        Response with proper multivariate shape

    Example:
        Point forecast:
            Input shape:  (batch*features=6, horizon=24, features=1)
            Output shape: (batch=2, horizon=24, features=3)

        Quantile forecast:
            Input shape:  (batch*features=6, horizon=24, quantiles=5, features=1)
            Output shape: (batch=2, horizon=24, quantiles=5, features=3)
    """
    modified_response = ForecastResponse(metadata=response.metadata)

    # Reshape point predictions if present
    if response.point is not None:
        # Input: (batch*features, horizon, 1)
        # Output: (batch, horizon, features)
        batch_times_features, horizon, _ = response.point.shape

        # Reshape to (batch, features, horizon, 1)
        reshaped = response.point.reshape(original_batch_size, num_features, horizon, 1)
        # Transpose to (batch, horizon, features, 1)
        transposed = reshaped.transpose(0, 2, 1, 3)  # (batch, horizon, features, 1)
        # Squeeze last dimension to get (batch, horizon, features)
        modified_response.point = transposed.squeeze(-1)

    # Reshape quantile predictions if present
    if response.quantiles is not None:
        # Input: (batch*features, horizon, quantiles, 1)
        # Output: (batch, horizon, quantiles, features)
        batch_times_features, horizon, num_quantiles, _ = response.quantiles.shape

        # Reshape to (batch, features, horizon, quantiles, 1)
        reshaped = response.quantiles.reshape(original_batch_size, num_features, horizon, num_quantiles, 1)
        # Transpose to (batch, horizon, quantiles, features, 1)
        transposed = reshaped.transpose(0, 2, 3, 1, 4)  # (batch, horizon, quantiles, features, 1)
        # Squeeze last dimension to get (batch, horizon, quantiles, features)
        modified_response.quantiles = transposed.squeeze(-1)

    # Samples - keep as is for now (not common for FlowState/TiRex)
    if response.samples is not None:
        modified_response.samples = response.samples

    logger.debug(f"Reshaped univariate response: original_batch={original_batch_size}, features={num_features}")

    return modified_response


class ForecastClient:
    """High-level client for FAIM time-series forecasting.

    Provides a clean, type-safe API over the generated faim_client with:
    - Automatic Arrow serialization/deserialization
    - Comprehensive error handling with specific exception types
    - Request/response logging for observability
    - Support for both sync and async operations
    - Automatic model inference from request type

    Example:
        >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
        >>>
        >>> client = ForecastClient(base_url="https://api.faim.it.com")
        >>> request = Chronos2ForecastRequest(
        ...     x=data,
        ...     horizon=10,
        ...     quantiles=[0.1, 0.5, 0.9]
        ... )
        >>> response = client.forecast(request)  # Model inferred automatically
        >>> print(response.quantiles.shape)
    """

    def __init__(
        self,
        base_url: str = "https://api.faim.it.com",
        timeout: float = 60.0,
        verify_ssl: bool = True,
        api_key: str | None = None,
        **httpx_kwargs,
    ) -> None:
        """Initialize FAIM forecast client.

        Args:
            base_url: Base URL of FAIM inference API
            timeout: Request timeout in seconds. Default: 60s
            verify_ssl: Whether to verify SSL certificates. Default: True
            api_key: Optional API key for authentication. If provided, all requests
                     will include "Authorization: Bearer <api_key>" header. Default: None
            **httpx_kwargs: Additional arguments passed to httpx.Client
                           (e.g., headers, limits, proxies)

        Example:
            >>> # Without authentication
            >>> client = ForecastClient(base_url="https://api.example.com")

            >>> # With API key authentication
            >>> client = ForecastClient(
            ...     base_url="https://api.example.com",
            ...     api_key="your-secret-api-key"
            ... )
        """
        self.base_url = base_url
        timeout_obj = httpx.Timeout(timeout)

        if api_key:
            self._client = AuthenticatedClient(
                base_url=base_url,
                timeout=timeout_obj,
                verify_ssl=verify_ssl,
                token=api_key,
                prefix="Bearer",
                **httpx_kwargs,
            )
            logger.info(f"Initialized ForecastClient with authentication: base_url={base_url}, timeout={timeout}s")
        else:
            self._client = Client(
                base_url=base_url,
                timeout=timeout_obj,
                verify_ssl=verify_ssl,
                **httpx_kwargs,
            )
            logger.info(f"Initialized ForecastClient: base_url={base_url}, timeout={timeout}s")

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate time-series forecast (synchronous).

        The model is automatically inferred from the request type:
        - FlowStateForecastRequest → FlowState
        - Chronos2ForecastRequest → Chronos2
        - TiRexForecastRequest → TiRex

        Args:
            request: Model-specific forecast request (FlowStateForecastRequest,
                    Chronos2ForecastRequest, or TiRexForecastRequest)

        Returns:
            ForecastResponse with predictions and metadata

        Raises:
            AuthenticationError: If authentication fails (401, 403)
            InsufficientFundsError: If billing account balance is insufficient (402)
            ModelNotFoundError: If model or version doesn't exist (404)
            PayloadTooLargeError: If request exceeds size limit (413)
            ValidationError: If request parameters are invalid (422)
            RateLimitError: If rate limit exceeded (429)
            InternalServerError: If backend encounters error (500)
            ServiceUnavailableError: If service unavailable (503, 504)
            NetworkError: If network communication fails
            SerializationError: If request serialization or response deserialization fails
            TimeoutError: If request exceeds timeout
            APIError: For other API errors

        Example:
            >>> request = FlowStateForecastRequest(x=data, horizon=10)
            >>> response = client.forecast(request)
        """
        model = request.model_name
        logger.debug(
            f"Starting forecast: model={model}, version={request.model_version}, "
            f"x.shape={request.x.shape}, horizon={request.horizon}"
        )

        # Check if univariate transformation is needed
        transform_shape_info = None
        if _needs_univariate_transformation(request):
            request, transform_shape_info = _prepare_univariate_request(request)

        # Serialize request to Arrow format
        try:
            arrays, metadata = request.to_arrays_and_metadata()
            payload = serialize_to_arrow(arrays, metadata, compression=request.compression)
            logger.debug(f"Serialized request: {len(payload)} bytes, metadata={metadata}")

        except Exception as e:
            logger.exception("Request serialization failed")
            raise SerializationError(
                f"Failed to serialize request: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

        # Wrap payload in File object for generated client
        payload_file = File(payload=io.BytesIO(payload), mime_type="application/vnd.apache.arrow.stream")

        # Make API call
        try:
            response = forecast_v1_ts_forecast_model_name_model_version_post.sync_detailed(
                model_name=model,
                model_version=request.model_version,
                client=self._client,
                body=payload_file,
            )

        except KeyError as e:
            # Backend returned error response with unexpected format
            logger.error(f"Failed to parse error response: {e}")
            raise APIError(
                f"Server returned error with unexpected format (missing '{e}' field)",
                details={"model": str(model), "error_type": "ResponseParseError"},
            ) from e

        except httpx.TimeoutException as e:
            logger.error(f"Request timeout after {self._client._timeout}s")
            raise TimeoutError(
                f"Request exceeded timeout of {self._client._timeout}s",
                details={"model": str(model), "version": request.model_version},
            ) from e

        except httpx.NetworkError as e:
            logger.error(f"Network error: {e}")
            raise NetworkError(
                f"Network communication failed: {e}",
                details={"model": str(model), "base_url": self.base_url},
            ) from e

        except Exception as e:
            logger.exception("Unexpected error during API call")
            raise APIError(
                f"Unexpected error: {e}",
                details={"model": str(model), "error_type": type(e).__name__},
            ) from e

        # Handle non-200 responses with error contract
        if response.status_code != 200:
            error_response = _parse_error_response(response)

            # Build error message from ErrorResponse or fallback
            if error_response:
                message = error_response.message
                if error_response.detail:
                    message = f"{message}: {error_response.detail}"

                logger.error(
                    f"API error: status={response.status_code}, "
                    f"code={error_response.error_code}, "
                    f"message={error_response.message}, "
                    f"request_id={error_response.request_id}"
                )
            else:
                message = f"Request failed with status {response.status_code}"
                logger.error(f"API error: status={response.status_code}, no error_response")

            # Map HTTP status code to exception class
            if response.status_code in (401, 403):
                raise AuthenticationError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 402:
                raise InsufficientFundsError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 404:
                raise ModelNotFoundError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 413:
                raise PayloadTooLargeError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 422:
                raise ValidationError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 429:
                raise RateLimitError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 500:
                raise InternalServerError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code in (503, 504):
                raise ServiceUnavailableError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            else:
                # Fallback for unmapped status codes
                raise APIError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )

        # Deserialize successful response
        try:
            response_bytes = response.content
            logger.debug(f"Received response: {len(response_bytes)} bytes")

            arrays, metadata = deserialize_from_arrow(response_bytes)
            forecast_response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)

            # If univariate transformation was applied, reshape response back
            if transform_shape_info is not None:
                original_batch_size, num_features = transform_shape_info
                forecast_response = _reshape_univariate_response(forecast_response, original_batch_size, num_features)

            logger.info(f"Forecast successful: {forecast_response}")
            return forecast_response

        except Exception as e:
            logger.exception("Response deserialization failed")
            raise SerializationError(
                f"Failed to deserialize response: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

    async def forecast_async(self, request: ForecastRequest) -> ForecastResponse:
        """Generate time-series forecast (asynchronous).

        The model is automatically inferred from the request type:
        - FlowStateForecastRequest → FlowState
        - Chronos2ForecastRequest → Chronos2
        - TiRexForecastRequest → TiRex

        Args:
            request: Model-specific forecast request (FlowStateForecastRequest,
                    Chronos2ForecastRequest, or TiRexForecastRequest)

        Returns:
            ForecastResponse with predictions and metadata

        Raises:
            Same exceptions as forecast()

        Example:
            >>> request = Chronos2ForecastRequest(x=data, horizon=10)
            >>> response = await client.forecast_async(request)
        """
        model = request.model_name
        logger.debug(f"Starting async forecast: model={model}, version={request.model_version}")

        # Check if univariate transformation is needed
        transform_shape_info = None
        if _needs_univariate_transformation(request):
            request, transform_shape_info = _prepare_univariate_request(request)

        # Serialize request
        try:
            arrays, metadata = request.to_arrays_and_metadata()
            payload = serialize_to_arrow(arrays, metadata, compression=request.compression)
            logger.debug(f"Serialized request: {len(payload)} bytes")

        except Exception as e:
            logger.exception("Request serialization failed")
            raise SerializationError(
                f"Failed to serialize request: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

        # Wrap payload in File object for generated client
        payload_file = File(payload=io.BytesIO(payload), mime_type="application/vnd.apache.arrow.stream")

        # Make async API call
        try:
            response = await forecast_v1_ts_forecast_model_name_model_version_post.asyncio_detailed(
                model_name=model,
                model_version=request.model_version,
                client=self._client,
                body=payload_file,
            )

        except KeyError as e:
            # Backend returned error response with unexpected format
            logger.error(f"Failed to parse error response: {e}")
            raise APIError(
                f"Server returned error with unexpected format (missing '{e}' field)",
                details={"model": str(model), "error_type": "ResponseParseError"},
            ) from e

        except httpx.TimeoutException as e:
            logger.error(f"Request timeout after {self._client._timeout}s")
            raise TimeoutError(
                f"Request exceeded timeout of {self._client._timeout}s",
                details={"model": str(model), "version": request.model_version},
            ) from e

        except httpx.NetworkError as e:
            logger.error(f"Network error: {e}")
            raise NetworkError(
                f"Network communication failed: {e}",
                details={"model": str(model), "base_url": self.base_url},
            ) from e

        except Exception as e:
            logger.exception("Unexpected error during async API call")
            raise APIError(
                f"Unexpected error: {e}",
                details={"model": str(model), "error_type": type(e).__name__},
            ) from e

        # Handle non-200 responses with error contract (same as sync)
        if response.status_code != 200:
            error_response = _parse_error_response(response)

            # Build error message from ErrorResponse or fallback
            if error_response:
                message = error_response.message
                if error_response.detail:
                    message = f"{message}: {error_response.detail}"

                logger.error(
                    f"API error: status={response.status_code}, "
                    f"code={error_response.error_code}, "
                    f"message={error_response.message}, "
                    f"request_id={error_response.request_id}"
                )
            else:
                message = f"Request failed with status {response.status_code}"
                logger.error(f"API error: status={response.status_code}, no error_response")

            # Map HTTP status code to exception class
            if response.status_code in (401, 403):
                raise AuthenticationError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 402:
                raise InsufficientFundsError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 404:
                raise ModelNotFoundError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 413:
                raise PayloadTooLargeError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 422:
                raise ValidationError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 429:
                raise RateLimitError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code == 500:
                raise InternalServerError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            elif response.status_code in (503, 504):
                raise ServiceUnavailableError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )
            else:
                # Fallback for unmapped status codes
                raise APIError(
                    message=message,
                    status_code=response.status_code,
                    error_response=error_response,
                )

        # Deserialize response
        try:
            response_bytes = response.content
            logger.debug(f"Received response: {len(response_bytes)} bytes")

            arrays, metadata = deserialize_from_arrow(response_bytes)
            forecast_response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)

            # If univariate transformation was applied, reshape response back
            if transform_shape_info is not None:
                original_batch_size, num_features = transform_shape_info
                forecast_response = _reshape_univariate_response(forecast_response, original_batch_size, num_features)

            logger.info(f"Async forecast successful: {forecast_response}")
            return forecast_response

        except Exception as e:
            logger.exception("Response deserialization failed")
            raise SerializationError(
                f"Failed to deserialize response: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

    def close(self) -> None:
        """Close underlying HTTP client and release resources.

        Should be called when the client is no longer needed to properly
        release connection pool resources. Alternatively, use the client
        as a context manager which handles cleanup automatically.

        Example:
            >>> client = ForecastClient(base_url="https://api.example.com")
            >>> try:
            ...     response = client.forecast(model, request)
            ... finally:
            ...     client.close()
        """
        if hasattr(self._client, "_client") and self._client._client:
            self._client._client.close()
        logger.debug("ForecastClient closed")

    async def aclose(self) -> None:
        """Close underlying async HTTP client and release resources.

        Async equivalent of close(). Should be called when the async client
        is no longer needed. Alternatively, use the client as an async
        context manager which handles cleanup automatically.

        Example:
            >>> client = ForecastClient(base_url="https://api.example.com")
            >>> try:
            ...     response = await client.forecast_async(model, request)
            ... finally:
            ...     await client.aclose()
        """
        if hasattr(self._client, "_async_client") and self._client._async_client:
            await self._client._async_client.aclose()
        logger.debug("Async ForecastClient closed")

    def __enter__(self) -> "ForecastClient":
        """Enter sync context manager.

        Enables using the client with Python's 'with' statement for
        automatic resource cleanup.

        Returns:
            The client instance

        Example:
            >>> with ForecastClient(base_url="https://api.example.com") as client:
            ...     response = client.forecast(request)
            ...     # Client automatically closed on exit
        """
        return self

    def __exit__(self, *args) -> None:
        """Exit sync context manager and release resources.

        Automatically called when exiting a 'with' block. Ensures proper
        cleanup of HTTP connections.

        Args:
            *args: Exception information (exc_type, exc_value, traceback) if an
                   exception occurred within the with block
        """
        self.close()

    async def __aenter__(self) -> "ForecastClient":
        """Enter async context manager.

        Enables using the client with Python's 'async with' statement for
        automatic resource cleanup in async code.

        Returns:
            The client instance

        Example:
            >>> async with ForecastClient(base_url="https://api.example.com") as client:
            ...     response = await client.forecast_async(request)
            ...     # Client automatically closed on exit
        """
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context manager and release resources.

        Automatically called when exiting an 'async with' block. Ensures proper
        cleanup of HTTP connections.

        Args:
            *args: Exception information (exc_type, exc_value, traceback) if an
                   exception occurred within the async with block
        """
        await self.aclose()
