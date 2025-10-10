"""FAIM SDK - Production-ready Python client for FAIM time-series forecasting.

This SDK provides a high-level, type-safe interface for interacting with the
FAIM inference platform for foundation AI models on structured data.
"""

from .client import ForecastClient
from .exceptions import (
    APIError,
    ConfigurationError,
    FAIMError,
    InternalServerError,
    ModelNotFoundError,
    NetworkError,
    PayloadTooLargeError,
    SerializationError,
    TimeoutError,
    ValidationError,
)
from .models import (
    FlowStateForecastRequest,
    ForecastRequest,
    ForecastResponse,
    ToToForecastRequest,
)

__all__ = [
    # Client
    "ForecastClient",
    # Request models
    "ForecastRequest",
    "ToToForecastRequest",
    "FlowStateForecastRequest",
    # Response model
    "ForecastResponse",
    # Exceptions
    "FAIMError",
    "APIError",
    "SerializationError",
    "ModelNotFoundError",
    "PayloadTooLargeError",
    "ValidationError",
    "InternalServerError",
    "NetworkError",
    "TimeoutError",
    "ConfigurationError",
]

__version__ = "0.1.0"
