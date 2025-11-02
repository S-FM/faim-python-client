"""FAIM SDK - Production-ready Python client for FAIM time-series forecasting.

This SDK provides a high-level, type-safe interface for interacting with the
FAIM inference platform for foundation AI models on structured data.
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
    OutputType,
    TiRexForecastRequest,
)

__all__ = [
    # Client
    "ForecastClient",
    # Request models
    "ForecastRequest",
    "FlowStateForecastRequest",
    "Chronos2ForecastRequest",
    "TiRexForecastRequest",
    # Response model
    "ForecastResponse",
    # Type aliases
    "OutputType",
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

__version__ = "0.2.0"
