"""FAIM SDK - Production-ready Python client for FAIM time-series forecasting.

This SDK provides a high-level, type-safe interface for interacting with the
FAIM inference platform for foundation AI models on structured data.

Quick Start
-----------

Install the SDK:

    pip install faim-sdk

Basic usage:

    >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
    >>> import numpy as np
    >>>
    >>> # Initialize client
    >>> client = ForecastClient(
    ...     base_url="https://api.faim.it.com",
    ...     api_key="your-api-key"
    ... )
    >>>
    >>> # Create forecast request
    >>> data = np.random.rand(32, 100, 1)  # (batch_size, seq_len, features)
    >>> request = Chronos2ForecastRequest(
    ...     x=data,
    ...     horizon=10,
    ...     output_type="quantiles",
    ...     quantiles=[0.1, 0.5, 0.9]
    ... )
    >>>
    >>> # Generate forecast - model inferred automatically
    >>> response = client.forecast(request)
    >>> print(response.quantiles.shape)  # (32, 10, 3)

Main Components
---------------

ForecastClient:
    High-level client for making forecast requests with automatic
    serialization, error handling, and observability.

Request Models:
    - FlowStateForecastRequest: For FlowState model with scaling options
    - Chronos2ForecastRequest: For Amazon Chronos 2.0 LLM-based forecasting
    - TiRexForecastRequest: For TiRex transformer-based forecasting

ForecastResponse:
    Contains prediction outputs (point, quantiles, samples) and metadata.

Evaluation Tools:
    The faim_sdk.eval subpackage provides metrics and visualization:
    - Metrics: mse, mase, crps_from_quantiles
    - Visualization: plot_forecast (requires: pip install faim-sdk[viz])

    >>> from faim_sdk.eval import mse, plot_forecast
    >>> mse_score = mse(test_data, response.point)
    >>> fig, ax = plot_forecast(train_data[0], response.point[0], test_data[0])

Exceptions:
    - AuthenticationError: API key issues (401, 403)
    - ValidationError: Invalid request parameters (422)
    - RateLimitError: Rate limit exceeded (429)
    - ServiceUnavailableError: Temporary service issues (503, 504)
    - And more... (see exceptions module for full hierarchy)

Error Handling
--------------

The SDK uses machine-readable error codes for programmatic error handling:

    >>> from faim_sdk import ValidationError, ErrorCode, Chronos2ForecastRequest
    >>>
    >>> try:
    ...     request = Chronos2ForecastRequest(x=data, horizon=10, quantiles=[0.1, 0.5, 0.9])
    ...     response = client.forecast(request)
    >>> except ValidationError as e:
    ...     if e.error_code == ErrorCode.INVALID_SHAPE:
    ...         print(f"Shape error: {e.error_response.detail}")
    ...     print(f"Request ID: {e.error_response.request_id}")

For more information, see the individual module documentation.
"""

from faim_client.models.error_code import ErrorCode

from .client import ForecastClient
from .exceptions import (
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
from .models import (
    Chronos2ForecastRequest,
    FlowStateForecastRequest,
    ForecastRequest,
    ForecastResponse,
    LimiXPredictRequest,
    LimiXPredictResponse,
    OutputType,
    TaskType,
    TiRexForecastRequest,
)
from .tabular_client import TabularClient

__all__ = [
    # Clients
    "ForecastClient",
    "TabularClient",
    # Forecast request models
    "ForecastRequest",
    "FlowStateForecastRequest",
    "Chronos2ForecastRequest",
    "TiRexForecastRequest",
    # Forecast response model
    "ForecastResponse",
    # Tabular request/response models
    "LimiXPredictRequest",
    "LimiXPredictResponse",
    # Type aliases
    "OutputType",
    "TaskType",
    # Error codes (for programmatic error handling)
    "ErrorCode",
    # Exceptions
    "FAIMError",
    "APIError",
    "AuthenticationError",
    "InsufficientFundsError",
    "RateLimitError",
    "SerializationError",
    "ModelNotFoundError",
    "PayloadTooLargeError",
    "ValidationError",
    "InternalServerError",
    "ServiceUnavailableError",
    "NetworkError",
    "TimeoutError",
    "ConfigurationError",
]

__version__ = "0.5.0"
