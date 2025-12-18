"""Request and response models for FAIM SDK.

Provides type-safe interfaces for forecast requests and responses with
model-specific parameter classes.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from faim_client.models import ModelName

# Type aliases for output types
OutputType = Literal["point", "quantiles", "samples"]
TaskType = Literal["Classification", "Regression"]


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


@dataclass
class LimiXPredictRequest:
    """Prediction request for LimiX tabular inference model.

    LimiX - Foundation model for tabular classification and regression.
    Supports retrieval-augmented inference for improved accuracy on small datasets.

    Example:
        >>> import numpy as np
        >>> X_train = np.random.randn(100, 10).astype(np.float32)
        >>> y_train = np.random.randint(0, 2, 100).astype(np.float32)
        >>> X_test = np.random.randn(20, 10).astype(np.float32)
        >>> request = LimiXPredictRequest(
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     task_type="Classification"
        ... )
    """

    _model_name: ClassVar[ModelName] = ModelName.LIMIX

    X_train: np.ndarray
    """Training features. Shape: (n_train_samples, n_features)"""

    y_train: np.ndarray
    """Training labels. Shape: (n_train_samples,) or (n_train_samples, n_targets)"""

    X_test: np.ndarray
    """Test features for prediction. Shape: (n_test_samples, n_features)"""

    task_type: TaskType
    """Task type: 'Classification' or 'Regression' (case-sensitive)"""

    model_version: str = "1"
    """Model version to use. Default: '1'"""

    use_retrieval: bool = False
    """Enable retrieval-augmented inference (slower but potentially more accurate)"""

    compression: str | None = "zstd"
    """Arrow compression algorithm. Default: 'zstd'"""

    @property
    def model_name(self) -> ModelName:
        """Get the model name for this request type.

        Returns:
            ModelName enum value (always ModelName.LIMIX)
        """
        return self._model_name

    def __post_init__(self) -> None:
        """Validate LimiX-specific parameters.

        Raises:
            TypeError: If arrays are not numpy ndarrays
            ValueError: If array shapes are invalid or incompatible
        """
        # Validate array types
        if not isinstance(self.X_train, np.ndarray):
            raise TypeError(f"X_train must be numpy.ndarray, got {type(self.X_train).__name__}")
        if not isinstance(self.y_train, np.ndarray):
            raise TypeError(f"y_train must be numpy.ndarray, got {type(self.y_train).__name__}")
        if not isinstance(self.X_test, np.ndarray):
            raise TypeError(f"X_test must be numpy.ndarray, got {type(self.X_test).__name__}")

        # Validate array dimensions
        if self.X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D (n_samples, n_features), got shape {self.X_train.shape}")
        if self.X_test.ndim != 2:
            raise ValueError(f"X_test must be 2D (n_samples, n_features), got shape {self.X_test.shape}")

        # Validate feature dimension match
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(
                f"X_train and X_test must have same number of features. "
                f"Got X_train: {self.X_train.shape[1]}, X_test: {self.X_test.shape[1]}"
            )

        # Validate y_train shape matches X_train samples
        if self.y_train.ndim == 1:
            n_train_labels = self.y_train.shape[0]
        elif self.y_train.ndim == 2:
            n_train_labels = self.y_train.shape[0]
        else:
            raise ValueError(f"y_train must be 1D or 2D, got shape {self.y_train.shape}")

        if n_train_labels != self.X_train.shape[0]:
            raise ValueError(
                f"y_train must have same number of samples as X_train. "
                f"Got y_train: {n_train_labels}, X_train: {self.X_train.shape[0]}"
            )

        # Validate task_type
        if self.task_type not in ("Classification", "Regression"):
            raise ValueError(f"task_type must be 'Classification' or 'Regression', got '{self.task_type}'")

    def to_arrays_and_metadata(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Convert LimiX request to Arrow-compatible format.

        Large arrays are placed in the arrays dict (sent as Arrow columns).
        Small parameters are placed in metadata (sent in Arrow schema).

        Returns:
            Tuple of (arrays dict, metadata dict) ready for Arrow serialization
        """
        arrays = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
        }

        metadata = {
            "task_type": self.task_type,
            "use_retrieval": self.use_retrieval,
        }

        return arrays, metadata


@dataclass
class LimiXPredictResponse:
    """Type-safe LimiX prediction response.

    Contains predictions, optional class probabilities, and metadata.

    Attributes:
        predictions: Model predictions. Shape: (n_test_samples,)
        metadata: Response metadata from backend (model_name, task_type, token_count, etc.)
        probabilities: Class probabilities for classification. Shape: (n_test_samples, n_classes)
                      None for regression tasks.
    """

    predictions: np.ndarray
    """Predictions. Shape: (n_test_samples,)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Response metadata from backend (e.g., model_name, task_type, token_count)"""

    probabilities: np.ndarray | None = None
    """Class probabilities (classification only). Shape: (n_test_samples, n_classes)"""

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
    ) -> "LimiXPredictResponse":
        """Construct response from deserialized Arrow data.

        Args:
            arrays: Dictionary of numpy arrays from Arrow deserialization
            metadata: Metadata dictionary from Arrow schema

        Returns:
            LimiXPredictResponse instance

        Raises:
            ValueError: If predictions array is missing
        """
        predictions = arrays.get("predictions")
        if predictions is None:
            raise ValueError(f"Response missing 'predictions' array. Available: {list(arrays.keys())}")

        probabilities = arrays.get("probabilities")  # Optional for classification

        return cls(
            predictions=predictions,
            metadata=metadata,
            probabilities=probabilities,
        )

    def __repr__(self) -> str:
        """Return string representation of LimiX response.

        Returns:
            Human-readable string showing outputs and shapes
        """
        outputs = [f"predictions.shape={self.predictions.shape}"]
        if self.probabilities is not None:
            outputs.append(f"probabilities.shape={self.probabilities.shape}")

        outputs_str = ", ".join(outputs)
        return f"LimiXPredictResponse(outputs=[{outputs_str}], metadata={self.metadata})"
