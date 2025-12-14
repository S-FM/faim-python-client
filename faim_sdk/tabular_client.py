"""FAIM SDK client for tabular machine learning inference.

Provides high-level, type-safe API for LimiX tabular classification and regression
with automatic serialization, error handling, and observability.
"""

import io
import json
import logging

import httpx

from faim_client import AuthenticatedClient, Client
from faim_client.api.tabular import predict_tabular_v1_tabular_predict_model_name_model_version_post
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
from .models import LimiXPredictRequest, LimiXPredictResponse
from .utils import deserialize_from_arrow_tabular, serialize_to_arrow_tabular

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


class TabularClient:
    """High-level client for FAIM tabular inference (LimiX).

    Provides a clean, type-safe API over the generated faim_client with:
    - Automatic Arrow serialization/deserialization
    - Comprehensive error handling with specific exception types
    - Request/response logging for observability
    - Support for both sync and async operations
    - Automatic model inference from request type

    Example:
        >>> from faim_sdk import TabularClient, LimiXPredictRequest
        >>> import numpy as np
        >>>
        >>> client = TabularClient(base_url="https://api.faim.it.com")
        >>> X_train = np.random.randn(100, 10).astype(np.float32)
        >>> y_train = np.random.randint(0, 2, 100).astype(np.float32)
        >>> X_test = np.random.randn(20, 10).astype(np.float32)
        >>> request = LimiXPredictRequest(
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     task_type="Classification"
        ... )
        >>> response = client.predict(request)  # Model inferred automatically
        >>> print(response.predictions.shape)
    """

    def __init__(
        self,
        base_url: str = "https://api.faim.it.com",
        timeout: float = 60.0,
        verify_ssl: bool = True,
        api_key: str | None = None,
        **httpx_kwargs,
    ) -> None:
        """Initialize FAIM tabular client.

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
            >>> client = TabularClient(base_url="https://api.example.com")

            >>> # With API key authentication
            >>> client = TabularClient(
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
            logger.info(f"Initialized TabularClient with authentication: base_url={base_url}, timeout={timeout}s")
        else:
            self._client = Client(
                base_url=base_url,
                timeout=timeout_obj,
                verify_ssl=verify_ssl,
                **httpx_kwargs,
            )
            logger.info(f"Initialized TabularClient: base_url={base_url}, timeout={timeout}s")

    def predict(self, request: LimiXPredictRequest) -> LimiXPredictResponse:
        """Generate tabular predictions (synchronous).

        Uses the LimiX foundation model for classification or regression on tabular data.

        Args:
            request: LimiX prediction request with training data, labels, and test data

        Returns:
            LimiXPredictResponse with predictions and metadata

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
            >>> request = LimiXPredictRequest(
            ...     X_train=X_train, y_train=y_train, X_test=X_test,
            ...     task_type="Classification"
            ... )
            >>> response = client.predict(request)
            >>> print(response.predictions.shape)  # (n_test_samples,)
            >>> print(response.probabilities.shape)  # (n_test_samples, n_classes)
        """
        model = request.model_name
        logger.debug(
            f"Starting tabular prediction: model={model}, version={request.model_version}, "
            f"X_train.shape={request.X_train.shape}, X_test.shape={request.X_test.shape}, "
            f"task_type={request.task_type}"
        )

        # Serialize request to Arrow format
        try:
            arrays, metadata = request.to_arrays_and_metadata()
            payload = serialize_to_arrow_tabular(arrays, metadata, compression=request.compression)
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
            response = predict_tabular_v1_tabular_predict_model_name_model_version_post.sync_detailed(
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

            arrays, metadata = deserialize_from_arrow_tabular(response_bytes)
            limix_response = LimiXPredictResponse.from_arrays_and_metadata(arrays, metadata)

            logger.info(f"Prediction successful: {limix_response}")
            return limix_response

        except Exception as e:
            logger.exception("Response deserialization failed")
            raise SerializationError(
                f"Failed to deserialize response: {e}",
                details={"model": str(model), "error": str(e)},
            ) from e

    async def predict_async(self, request: LimiXPredictRequest) -> LimiXPredictResponse:
        """Generate tabular predictions (asynchronous).

        Uses the LimiX foundation model for classification or regression on tabular data.

        Args:
            request: LimiX prediction request with training data, labels, and test data

        Returns:
            LimiXPredictResponse with predictions and metadata

        Raises:
            Same exceptions as predict()

        Example:
            >>> request = LimiXPredictRequest(
            ...     X_train=X_train, y_train=y_train, X_test=X_test,
            ...     task_type="Regression"
            ... )
            >>> response = await client.predict_async(request)
        """
        model = request.model_name
        logger.debug(f"Starting async prediction: model={model}, version={request.model_version}")

        # Serialize request
        try:
            arrays, metadata = request.to_arrays_and_metadata()
            payload = serialize_to_arrow_tabular(arrays, metadata, compression=request.compression)
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
            response = await predict_tabular_v1_tabular_predict_model_name_model_version_post.asyncio_detailed(
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

            arrays, metadata = deserialize_from_arrow_tabular(response_bytes)
            limix_response = LimiXPredictResponse.from_arrays_and_metadata(arrays, metadata)

            logger.info(f"Async prediction successful: {limix_response}")
            return limix_response

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
            >>> client = TabularClient(base_url="https://api.example.com")
            >>> try:
            ...     response = client.predict(request)
            ... finally:
            ...     client.close()
        """
        if hasattr(self._client, "_client") and self._client._client:
            self._client._client.close()
        logger.debug("TabularClient closed")

    async def aclose(self) -> None:
        """Close underlying async HTTP client and release resources.

        Async equivalent of close(). Should be called when the async client
        is no longer needed. Alternatively, use the client as an async
        context manager which handles cleanup automatically.

        Example:
            >>> client = TabularClient(base_url="https://api.example.com")
            >>> try:
            ...     response = await client.predict_async(request)
            ... finally:
            ...     await client.aclose()
        """
        if hasattr(self._client, "_async_client") and self._client._async_client:
            await self._client._async_client.aclose()
        logger.debug("Async TabularClient closed")

    def __enter__(self) -> "TabularClient":
        """Enter sync context manager.

        Enables using the client with Python's 'with' statement for
        automatic resource cleanup.

        Returns:
            The client instance

        Example:
            >>> with TabularClient(base_url="https://api.example.com") as client:
            ...     response = client.predict(request)
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

    async def __aenter__(self) -> "TabularClient":
        """Enter async context manager.

        Enables using the client with Python's 'async with' statement for
        automatic resource cleanup in async code.

        Returns:
            The client instance

        Example:
            >>> async with TabularClient(base_url="https://api.example.com") as client:
            ...     response = await client.predict_async(request)
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
