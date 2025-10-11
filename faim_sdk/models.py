"""Request and response models for FAIM SDK.

Provides type-safe interfaces for forecast requests and responses with
model-specific parameter classes.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np


@dataclass
class ForecastRequest:
    """Base forecast request with common parameters.

    This is the base class for all model-specific forecast requests.
    Use model-specific subclasses (ToToForecastRequest, FlowStateForecastRequest)
    for better type safety and IDE support.
    """

    x: np.ndarray
    """Time series data. Shape: (batch_size, sequence_length, features) or (sequence_length, features)"""

    horizon: int
    """Forecast horizon length (number of time steps to predict)"""

    model_version: str = "1"
    """Model version to use for inference. Default: '1'"""

    compression: Optional[str] = "zstd"
    """Arrow compression algorithm. Options: 'zstd', 'lz4', None. Default: 'zstd'"""

    def __post_init__(self) -> None:
        """Validate common parameters."""
        if not isinstance(self.x, np.ndarray):
            raise TypeError(f"x must be numpy.ndarray, got {type(self.x).__name__}")

        if self.x.size == 0:
            raise ValueError("x cannot be empty")

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert request to Arrow-compatible arrays and metadata.

        Large arrays are placed in the arrays dict (sent as Arrow columns).
        Small parameters are placed in metadata (sent in Arrow schema).

        Returns:
            Tuple of (arrays dict, metadata dict)
        """
        # Base arrays - always include x
        arrays: dict[str, np.ndarray] = {"x": self.x}

        # Base metadata - always include horizon
        metadata: dict[str, Any] = {"horizon": self.horizon}

        return arrays, metadata


@dataclass
class ToToForecastRequest(ForecastRequest):
    """Forecast request for ToTo model with probabilistic forecasting support.

    ToTo supports multi-series forecasting with padding masks and
    probabilistic predictions via sampling or quantiles.
    """

    padding_mask: Optional[np.ndarray] = None
    """Padding mask for variable-length sequences. Shape: same as x.
    1 for valid timesteps, 0 for padding."""

    id_mask: Optional[np.ndarray] = None
    """Identifier mask for multi-series forecasting. Shape: (batch_size,).
    Each unique ID represents a different time series."""

    num_samples: Optional[int] = None
    """Number of forecast samples for probabilistic predictions.
    If set, returns sample-based distribution."""

    quantiles: Optional[list[float]] = None
    """Quantile levels for probabilistic forecasting.
    Example: [0.1, 0.5, 0.9] for 10th, 50th (median), 90th percentiles."""

    def __post_init__(self) -> None:
        """Validate ToTo-specific parameters."""
        super().__post_init__()

        if self.padding_mask is not None:
            if not isinstance(self.padding_mask, np.ndarray):
                raise TypeError("padding_mask must be numpy.ndarray")
            if self.padding_mask.shape != self.x.shape:
                raise ValueError(f"padding_mask shape {self.padding_mask.shape} must match x shape {self.x.shape}")

        if self.id_mask is not None:
            if not isinstance(self.id_mask, np.ndarray):
                raise TypeError("id_mask must be numpy.ndarray")

        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

        if self.quantiles is not None:
            if not all(0 <= q <= 1 for q in self.quantiles):
                raise ValueError(f"quantiles must be in [0, 1], got {self.quantiles}")

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert ToTo request to Arrow format."""
        arrays, metadata = super().to_arrays_and_metadata()

        # Add ToTo-specific arrays (large data)
        if self.padding_mask is not None:
            arrays["padding_mask"] = self.padding_mask
        if self.id_mask is not None:
            arrays["id_mask"] = self.id_mask

        # Add ToTo-specific metadata (small parameters)
        if self.num_samples is not None:
            metadata["num_samples"] = self.num_samples
        if self.quantiles is not None:
            metadata["quantiles"] = self.quantiles

        return arrays, metadata


@dataclass
class FlowStateForecastRequest(ForecastRequest):
    """Forecast request for FlowState model with scaling and prediction type control.

    FlowState is optimized for point forecasts with optional scaling
    and different prediction modes.
    """

    scale_factor: Optional[float] = None
    """Scaling factor for normalization/denormalization.
    Applied to inputs before inference and outputs after inference."""

    prediction_type: Optional[Literal["point", "mean", "median"]] = None
    """Prediction type for FlowState model. Options: 'point', 'mean', 'median'."""

    def __post_init__(self) -> None:
        """Validate FlowState-specific parameters."""
        super().__post_init__()

        if self.scale_factor is not None and self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert FlowState request to Arrow format."""
        arrays, metadata = super().to_arrays_and_metadata()

        # Add FlowState-specific metadata
        if self.scale_factor is not None:
            metadata["scale_factor"] = self.scale_factor
        if self.prediction_type is not None:
            metadata["prediction_type"] = self.prediction_type

        return arrays, metadata


@dataclass
class ForecastResponse:
    """Type-safe forecast response.

    Contains predictions and metadata from backend inference.
    """

    predictions: np.ndarray
    """Forecast predictions. Shape depends on model and parameters.
    Typical shape: (batch_size, horizon) or (batch_size, horizon, features)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Response metadata from backend (e.g., model_version, inference_time_ms)"""

    # Optional model-specific outputs
    quantiles: Optional[np.ndarray] = None
    """Quantile predictions if requested. Shape: (batch_size, horizon, num_quantiles)"""

    samples: Optional[np.ndarray] = None
    """Forecast samples if num_samples was specified. Shape: (batch_size, horizon, num_samples)"""

    uncertainty: Optional[np.ndarray] = None
    """Uncertainty estimates if available. Shape: (batch_size, horizon)"""

    @classmethod
    def from_arrays_and_metadata(cls, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> "ForecastResponse":
        """Construct response from deserialized Arrow data.

        Args:
            arrays: Dictionary of numpy arrays from Arrow deserialization
            metadata: Metadata dictionary from Arrow schema

        Returns:
            ForecastResponse instance

        Raises:
            ValueError: If predictions array is missing
        """
        # Primary output - backend returns: point, quantiles, or samples
        # Use point as primary prediction (most common for FlowState)
        predictions = arrays.get("point")

        # If no point predictions, check if we have quantiles or samples as alternatives
        if predictions is None:
            predictions = arrays.get("quantiles")
        if predictions is None:
            predictions = arrays.get("samples")

        if predictions is None:
            raise ValueError(f"Response missing predictions array. Available keys: {list(arrays.keys())}")

        # Optional outputs (model-specific)
        quantiles = arrays.get("quantiles")
        samples = arrays.get("samples")
        uncertainty = arrays.get("uncertainty")

        return cls(
            predictions=predictions,
            metadata=metadata,
            quantiles=quantiles,
            samples=samples,
            uncertainty=uncertainty,
        )

    def __repr__(self) -> str:
        optional_outputs = []
        if self.quantiles is not None:
            optional_outputs.append(f"quantiles.shape={self.quantiles.shape}")
        if self.samples is not None:
            optional_outputs.append(f"samples.shape={self.samples.shape}")
        if self.uncertainty is not None:
            optional_outputs.append(f"uncertainty.shape={self.uncertainty.shape}")

        optional_str = ", ".join(optional_outputs) if optional_outputs else "None"

        return (
            f"ForecastResponse("
            f"predictions.shape={self.predictions.shape}, "
            f"optional_outputs=[{optional_str}], "
            f"metadata_keys={list(self.metadata.keys())})"
        )
