"""Request and response models for FAIM SDK.

Provides type-safe interfaces for forecast requests and responses with
model-specific parameter classes.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from faim_client.models import ModelName

# Type alias for output types
OutputType = Literal["point", "quantiles", "samples"]


@dataclass
class ForecastRequest:
    """Base forecast request with common parameters.

    This is the base class for all model-specific forecast requests.
    Use model-specific subclasses (FlowStateForecastRequest, Chronos2ForecastRequest, TiRexForecastRequest)
    for better type safety and IDE support.
    """

    # Class variable to be overridden by subclasses
    _model_name: ClassVar[ModelName]

    x: np.ndarray
    """Time series data. Shape: (batch_size, sequence_length, features)"""

    horizon: int
    """Forecast horizon length (number of time steps to predict)"""

    model_version: str = "1"
    """Model version to use for inference. Default: '1'"""

    compression: str | None = "zstd"
    """Arrow compression algorithm. Options: 'zstd', 'lz4', None. Default: 'zstd'"""

    @property
    def model_name(self) -> ModelName:
        """Get the model name for this request type.

        Returns:
            ModelName enum value indicating which model to use

        Example:
            >>> request = Chronos2ForecastRequest(x=data, horizon=10)
            >>> print(request.model_name)  # ModelName.CHRONOS2
        """
        return self._model_name

    def __post_init__(self) -> None:
        """Validate common parameters.

        Automatically called after dataclass initialization to ensure
        all parameters meet requirements.

        Raises:
            TypeError: If x is not a numpy ndarray
            ValueError: If x is empty, not 3D, or horizon is non-positive
        """
        if not isinstance(self.x, np.ndarray):
            raise TypeError(f"x must be numpy.ndarray, got {type(self.x).__name__}")

        if self.x.size == 0:
            raise ValueError("x cannot be empty")

        # Ensure x is 3D: (batch_size, sequence_length, features)
        if self.x.ndim != 3:
            raise ValueError(
                f"x must be a 3D array with shape (batch_size, sequence_length, features), "
                f"got shape {self.x.shape} with {self.x.ndim} dimensions"
            )

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
class Chronos2ForecastRequest(ForecastRequest):
    """Forecast request for Chronos2 model.

    Amazon Chronos 2.0 - Large language model for time series forecasting.
    Supports point and quantile predictions.
    """

    _model_name: ClassVar[ModelName] = ModelName.CHRONOS2

    output_type: OutputType = "point"
    """Output type to return. Options: 'point', 'quantiles', 'samples'. Default: 'point'."""

    quantiles: list[float] | None = None
    """Quantile levels for probabilistic forecasting.
    Example: [0.1, 0.5, 0.9] for 10th, 50th (median), 90th percentiles.
    Only used when output_type='quantiles'."""

    def __post_init__(self) -> None:
        """Validate Chronos2-specific parameters.

        Automatically called after dataclass initialization to ensure
        quantiles are valid probability values.

        Raises:
            ValueError: If quantiles are not in the range [0.0, 1.0]
        """
        super().__post_init__()

        if self.quantiles is not None:
            if not all(0.0 <= q <= 1.0 for q in self.quantiles):
                raise ValueError(f"quantiles must be in [0.0, 1.0], got {self.quantiles}")

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert Chronos2 request to Arrow format.

        Separates request into large arrays (sent as Arrow columns) and
        small metadata parameters (sent in Arrow schema metadata).

        Returns:
            Tuple of (arrays dict, metadata dict) ready for Arrow serialization
        """
        arrays, metadata = super().to_arrays_and_metadata()

        # Add Chronos2-specific metadata (small parameters)
        metadata["output_type"] = self.output_type
        if self.quantiles is not None:
            metadata["quantiles"] = self.quantiles

        return arrays, metadata


@dataclass
class TiRexForecastRequest(ForecastRequest):
    """Forecast request for TiRex model.

    TiRex - Transformer-based time series forecasting.
    Supports point and quantile predictions.
    """

    _model_name: ClassVar[ModelName] = ModelName.TIREX

    output_type: OutputType = "point"
    """Output type to return. Options: 'point', 'quantiles', 'samples'. Default: 'point'."""

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert TiRex request to Arrow format.

        Separates request into large arrays (sent as Arrow columns) and
        small metadata parameters (sent in Arrow schema metadata).

        Returns:
            Tuple of (arrays dict, metadata dict) ready for Arrow serialization
        """
        arrays, metadata = super().to_arrays_and_metadata()

        metadata["output_type"] = self.output_type

        return arrays, metadata


@dataclass
class FlowStateForecastRequest(ForecastRequest):
    """Forecast request for FlowState model with scaling and prediction type control.

    FlowState is optimized for point forecasts with optional scaling
    and different prediction modes.
    """

    _model_name: ClassVar[ModelName] = ModelName.FLOWSTATE

    output_type: OutputType = "point"
    """Output type to return. Options: 'point', 'quantiles', 'samples'. Default: 'point'."""

    scale_factor: float | None = None
    """Scaling factor for normalization/denormalization.
    Applied to inputs before inference and outputs after inference."""

    prediction_type: Literal["mean", "median", "quantile"] | None = None
    """Prediction type for FlowState model.
    Options: 'mean', 'median' (requires output_type='point'),
             'quantile' (requires output_type='quantiles')."""

    def __post_init__(self) -> None:
        """Validate FlowState-specific parameters.

        Automatically called after dataclass initialization to ensure
        parameter validity and consistency between output_type and prediction_type.

        Raises:
            ValueError: If scale_factor is non-positive, or if output_type and
                       prediction_type are incompatible
        """
        super().__post_init__()

        if self.scale_factor is not None and self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

        # Validate prediction_type and output_type correspondence
        if self.prediction_type is not None:
            if self.prediction_type == "quantile":
                if self.output_type != "quantiles":
                    raise ValueError(
                        f"prediction_type='quantile' requires output_type='quantiles', got '{self.output_type}'"
                    )
            elif self.prediction_type in ("mean", "median"):
                if self.output_type != "point":
                    raise ValueError(
                        f"prediction_type='{self.prediction_type}' requires output_type='point', got '{self.output_type}'"
                    )
        elif self.output_type == "quantiles":
            self.prediction_type = "quantile"
        else:
            self.prediction_type = "median"

        # Validate output_type requires corresponding prediction_type
        if self.output_type == "quantiles" and self.prediction_type != "quantile":
            raise ValueError(
                f"output_type='quantiles' requires prediction_type='quantile', "
                f"got prediction_type='{self.prediction_type}'"
            )
        if self.output_type == "point" and self.prediction_type == "quantile":
            raise ValueError("output_type='point' conflicts with prediction_type='quantile'")

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert FlowState request to Arrow format.

        Separates request into large arrays (sent as Arrow columns) and
        small metadata parameters (sent in Arrow schema metadata).

        Returns:
            Tuple of (arrays dict, metadata dict) ready for Arrow serialization
        """
        arrays, metadata = super().to_arrays_and_metadata()

        # Add FlowState-specific metadata
        metadata["output_type"] = self.output_type
        if self.scale_factor is not None:
            metadata["scale_factor"] = self.scale_factor
        if self.prediction_type is not None:
            metadata["prediction_type"] = self.prediction_type

        return arrays, metadata


@dataclass
class ForecastResponse:
    """Type-safe forecast response.

    Contains outputs and metadata from backend inference.
    Backend returns one or more of: 'point', 'quantiles', 'samples'.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Response metadata from backend (e.g., model_name, model_version)"""

    # Backend outputs
    point: np.ndarray | None = None
    """Point predictions. Shape: (batch_size, horizon, features)"""

    quantiles: np.ndarray | None = None
    """Quantile predictions. Shape: (batch_size, horizon, num_quantiles, features)"""

    samples: np.ndarray | None = None
    """Sample predictions. Shape: (batch_size, horizon, num_samples)"""

    @classmethod
    def from_arrays_and_metadata(cls, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> "ForecastResponse":
        """Construct response from deserialized Arrow data.

        Args:
            arrays: Dictionary of numpy arrays from Arrow deserialization
            metadata: Metadata dictionary from Arrow schema

        Returns:
            ForecastResponse instance

        Raises:
            ValueError: If no output arrays found
        """
        # Extract backend outputs
        point = arrays.get("point")
        quantiles = arrays.get("quantiles")
        samples = arrays.get("samples")

        # Validate that at least one output is present
        if point is None and quantiles is None and samples is None:
            raise ValueError(f"Response missing output arrays. Available keys: {list(arrays.keys())}")

        return cls(
            metadata=metadata,
            point=point,
            quantiles=quantiles,
            samples=samples,
        )

    def __repr__(self) -> str:
        """Return string representation of forecast response.

        Returns:
            Human-readable string showing available outputs and their shapes
        """
        outputs = []
        if self.point is not None:
            outputs.append(f"point.shape={self.point.shape}")
        if self.quantiles is not None:
            outputs.append(f"quantiles.shape={self.quantiles.shape}")
        if self.samples is not None:
            outputs.append(f"samples.shape={self.samples.shape}")

        outputs_str = ", ".join(outputs) if outputs else "None"

        return f"ForecastResponse(outputs=[{outputs_str}], metadata={self.metadata})"
