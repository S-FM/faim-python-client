"""Production-ready FAIM SDK client for time-series forecasting.

Provides high-level, type-safe API with automatic serialization, error handling,
and observability.
"""

import io
import logging
from typing import Optional

import httpx

from faim_client import Client
from faim_client.api.forecast import forecast_v1_forecast_model_name_model_version_post
from faim_client.models import ModelName
from faim_client.types import File

from .exceptions import (
    APIError,
    InternalServerError,
    ModelNotFoundError,
    NetworkError,
    PayloadTooLargeError,
    SerializationError,
    TimeoutError,
    ValidationError,
)
from .models import ForecastRequest, ForecastResponse
from .utils import deserialize_from_arrow, serialize_to_arrow

logger = logging.getLogger(__name__)


class ForecastClient:
    """High-level client for FAIM time-series forecasting.

    Provides a clean, type-safe API over the generated faim_client with:
    - Automatic Arrow serialization/deserialization
    - Comprehensive error handling with specific exception types
    - Request/response logging for observability
    - Support for both sync and async operations

    Example:
        >>> from faim_sdk import ForecastClient, ToToForecastRequest
        >>> from faim_client.models import ModelName
        >>>
        >>> client = ForecastClient(base_url="https://api.example.com")
        >>> request = ToToForecastRequest(
        ...     x=data,
        ...     horizon=10,
        ...     quantiles=[0.1, 0.5, 0.9]
        ... )
        >>> response = client.forecast(ModelName.TOTO, request)
        >>> print(response.predictions.shape)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        verify_ssl: bool = True,
        **httpx_kwargs,
    ) -> None:
        """Initialize FAIM forecast client.

        Args:
            base_url: Base URL of FAIM inference API
            timeout: Request timeout in seconds. Default: 120s
            verify_ssl: Whether to verify SSL certificates. Default: True
            **httpx_kwargs: Additional arguments passed to httpx.Client
                           (e.g., headers, limits, proxies)

        Example:
            >>> client = ForecastClient(
            ...     base_url="https://api.example.com",
            ...     timeout=300.0,
            ...     httpx_args={"limits": httpx.Limits(max_connections=10)}
            ... )
        """
        self.base_url = base_url
        timeout_obj = httpx.Timeout(timeout)

        self._client = Client(
            base_url=base_url,
            timeout=timeout_obj,
            verify_ssl=verify_ssl,
            **httpx_kwargs,
        )

        logger.info(f"Initialized ForecastClient: base_url={base_url}, timeout={timeout}s")

    def forecast(
        self, model: ModelName, request: ForecastRequest
    ) -> ForecastResponse:
        """Generate time-series forecast (synchronous).

        Args:
            model: Model to use (ModelName.TOTO or ModelName.FLOWSTATE)
            request: Model-specific forecast request

        Returns:
            ForecastResponse with predictions and metadata

        Raises:
            SerializationError: If request serialization or response deserialization fails
            ModelNotFoundError: If model or version doesn't exist (404)
            PayloadTooLargeError: If request exceeds size limit (413)
            ValidationError: If request parameters are invalid (422)
            InternalServerError: If backend encounters error (500)
            NetworkError: If network communication fails
            TimeoutError: If request exceeds timeout
            APIError: For other API errors

        Example:
            >>> request = FlowStateForecastRequest(x=data, horizon=10)
            >>> response = client.forecast(ModelName.FLOWSTATE, request)
        """
        logger.debug(
            f"Starting forecast: model={model}, version={request.model_version}, "
            f"x.shape={request.x.shape}, horizon={request.horizon}"
        )

        # Serialize request to Arrow format
        try:
            arrays, metadata = request.to_arrays_and_metadata()
            payload = serialize_to_arrow(
                arrays, metadata, compression=request.compression
            )
            logger.debug(f"Serialized request: {len(payload)} bytes")

        except Exception as e:
            logger.exception("Request serialization failed")
            raise SerializationError(
                f"Failed to serialize request: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

        # Wrap payload in File object for generated client
        payload_file = File(
            payload=io.BytesIO(payload),
            mime_type="application/vnd.apache.arrow.stream"
        )

        # Make API call
        try:
            with self._client as client:
                response = forecast_v1_forecast_model_name_model_version_post.sync_detailed(
                    model_name=model,
                    model_version=request.model_version,
                    client=client,
                    body=payload_file,
                )

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

        # Handle HTTP errors
        if response.status_code == 404:
            logger.error(f"Model not found: {model}/{request.model_version}")
            raise ModelNotFoundError(
                f"Model {model}/{request.model_version} not found",
                status_code=404,
                response=response.parsed if hasattr(response, "parsed") else None,
            )

        elif response.status_code == 413:
            logger.error(f"Payload too large: {len(payload)} bytes")
            raise PayloadTooLargeError(
                f"Request payload too large ({len(payload)} bytes)",
                status_code=413,
                details={"payload_size": len(payload)},
            )

        elif response.status_code == 422:
            logger.error("Request validation failed")
            raise ValidationError(
                "Request parameters are invalid",
                status_code=422,
                response=response.parsed if hasattr(response, "parsed") else None,
            )

        elif response.status_code == 500:
            logger.error("Internal server error")
            raise InternalServerError(
                "Backend encountered an internal error",
                status_code=500,
                response=response.parsed if hasattr(response, "parsed") else None,
            )

        elif response.status_code != 200:
            logger.error(f"Unexpected status code: {response.status_code}")
            raise APIError(
                f"Unexpected status code: {response.status_code}",
                status_code=response.status_code,
                response=response.parsed if hasattr(response, "parsed") else None,
            )

        # Deserialize successful response
        try:
            response_bytes = response.content
            logger.debug(f"Received response: {len(response_bytes)} bytes")

            arrays, metadata = deserialize_from_arrow(response_bytes)
            forecast_response = ForecastResponse.from_arrays_and_metadata(
                arrays, metadata
            )

            logger.info(
                f"Forecast successful: predictions.shape={forecast_response.predictions.shape}"
            )
            return forecast_response

        except Exception as e:
            logger.exception("Response deserialization failed")
            raise SerializationError(
                f"Failed to deserialize response: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

    async def forecast_async(
        self, model: ModelName, request: ForecastRequest
    ) -> ForecastResponse:
        """Generate time-series forecast (asynchronous).

        Args:
            model: Model to use (ModelName.TOTO or ModelName.FLOWSTATE)
            request: Model-specific forecast request

        Returns:
            ForecastResponse with predictions and metadata

        Raises:
            Same exceptions as forecast()

        Example:
            >>> request = ToToForecastRequest(x=data, horizon=10)
            >>> response = await client.forecast_async(ModelName.TOTO, request)
        """
        logger.debug(
            f"Starting async forecast: model={model}, version={request.model_version}"
        )

        # Serialize request
        try:
            arrays, metadata = request.to_arrays_and_metadata()
            payload = serialize_to_arrow(
                arrays, metadata, compression=request.compression
            )
            logger.debug(f"Serialized request: {len(payload)} bytes")

        except Exception as e:
            logger.exception("Request serialization failed")
            raise SerializationError(
                f"Failed to serialize request: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

        # Wrap payload in File object for generated client
        payload_file = File(
            payload=io.BytesIO(payload),
            mime_type="application/vnd.apache.arrow.stream"
        )

        # Make async API call
        try:
            async with self._client as client:
                response = await forecast_v1_forecast_model_name_model_version_post.asyncio_detailed(
                    model_name=model,
                    model_version=request.model_version,
                    client=client,
                    body=payload_file,
                )

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

        # Handle HTTP errors (same as sync)
        if response.status_code == 404:
            raise ModelNotFoundError(
                f"Model {model}/{request.model_version} not found",
                status_code=404,
                response=response.parsed if hasattr(response, "parsed") else None,
            )
        elif response.status_code == 413:
            raise PayloadTooLargeError(
                f"Request payload too large ({len(payload)} bytes)",
                status_code=413,
                details={"payload_size": len(payload)},
            )
        elif response.status_code == 422:
            raise ValidationError(
                "Request parameters are invalid",
                status_code=422,
                response=response.parsed if hasattr(response, "parsed") else None,
            )
        elif response.status_code == 500:
            raise InternalServerError(
                "Backend encountered an internal error",
                status_code=500,
                response=response.parsed if hasattr(response, "parsed") else None,
            )
        elif response.status_code != 200:
            raise APIError(
                f"Unexpected status code: {response.status_code}",
                status_code=response.status_code,
                response=response.parsed if hasattr(response, "parsed") else None,
            )

        # Deserialize response
        try:
            response_bytes = response.content
            logger.debug(f"Received response: {len(response_bytes)} bytes")

            arrays, metadata = deserialize_from_arrow(response_bytes)
            forecast_response = ForecastResponse.from_arrays_and_metadata(
                arrays, metadata
            )

            logger.info(
                f"Async forecast successful: predictions.shape={forecast_response.predictions.shape}"
            )
            return forecast_response

        except Exception as e:
            logger.exception("Response deserialization failed")
            raise SerializationError(
                f"Failed to deserialize response: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

    def close(self) -> None:
        """Close underlying HTTP client and release resources."""
        if hasattr(self._client, "_client") and self._client._client:
            self._client._client.close()
        logger.debug("ForecastClient closed")

    async def aclose(self) -> None:
        """Close underlying async HTTP client and release resources."""
        if hasattr(self._client, "_async_client") and self._client._async_client:
            await self._client._async_client.aclose()
        logger.debug("Async ForecastClient closed")

    def __enter__(self) -> "ForecastClient":
        """Enter sync context manager."""
        return self

    def __exit__(self, *args) -> None:
        """Exit sync context manager."""
        self.close()

    async def __aenter__(self) -> "ForecastClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context manager."""
        await self.aclose()