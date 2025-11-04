"""Unit tests for faim_sdk.models module.

Tests request and response model validation, serialization, and type safety.
"""

import numpy as np
import pytest

from faim_client.models import ModelName
from faim_sdk.models import (
    Chronos2ForecastRequest,
    FlowStateForecastRequest,
    ForecastRequest,
    ForecastResponse,
    TiRexForecastRequest,
)


class TestForecastRequest:
    """Tests for base ForecastRequest class."""

    def test_cannot_instantiate_base_class_without_model_name(self):
        """Base class requires _model_name to be defined."""
        with pytest.raises(AttributeError):
            ForecastRequest(
                x=np.array([[1.0, 2.0], [3.0, 4.0]]),
                horizon=10,
            )

    def test_validation_requires_numpy_array(self):
        """x parameter must be numpy array."""
        with pytest.raises(TypeError, match="x must be numpy.ndarray"):
            Chronos2ForecastRequest(
                x=[[1.0, 2.0], [3.0, 4.0]],  # Python list, not ndarray
                horizon=10,
            )

    def test_validation_requires_non_empty_array(self):
        """x parameter cannot be empty."""
        with pytest.raises(ValueError, match="x cannot be empty"):
            Chronos2ForecastRequest(
                x=np.array([]),
                horizon=10,
            )

    def test_validation_requires_positive_horizon(self):
        """horizon must be positive."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            Chronos2ForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=0,
            )

    def test_validation_requires_positive_horizon_negative(self):
        """horizon cannot be negative."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            Chronos2ForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=-5,
            )


class TestChronos2ForecastRequest:
    """Tests for Chronos2ForecastRequest model."""

    def test_model_name_is_chronos2(self):
        """Model name should be CHRONOS2."""
        request = Chronos2ForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
        )
        assert request.model_name == ModelName.CHRONOS2

    def test_default_values(self):
        """Test default parameter values."""
        request = Chronos2ForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
        )
        assert request.model_version == "1"
        assert request.compression == "zstd"
        assert request.output_type == "point"
        assert request.quantiles is None

    def test_custom_values(self):
        """Test custom parameter values."""
        data = np.random.rand(32, 100, 1)
        request = Chronos2ForecastRequest(
            x=data,
            horizon=24,
            model_version="2.0",
            compression="lz4",
            output_type="quantiles",
            quantiles=[0.1, 0.5, 0.9],
        )
        assert request.horizon == 24
        assert request.model_version == "2.0"
        assert request.compression == "lz4"
        assert request.output_type == "quantiles"
        assert request.quantiles == [0.1, 0.5, 0.9]

    def test_quantiles_validation_requires_range_0_to_1(self):
        """Quantiles must be in [0.0, 1.0]."""
        with pytest.raises(ValueError, match="quantiles must be in"):
            Chronos2ForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                quantiles=[0.1, 0.5, 1.5],  # 1.5 is invalid
            )

    def test_quantiles_validation_negative_values(self):
        """Quantiles cannot be negative."""
        with pytest.raises(ValueError, match="quantiles must be in"):
            Chronos2ForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                quantiles=[-0.1, 0.5, 0.9],
            )

    def test_to_arrays_and_metadata(self):
        """Test conversion to Arrow format."""
        data = np.random.rand(32, 100, 1)
        request = Chronos2ForecastRequest(
            x=data,
            horizon=24,
            output_type="quantiles",
            quantiles=[0.1, 0.5, 0.9],
        )
        arrays, metadata = request.to_arrays_and_metadata()

        # Check arrays
        assert "x" in arrays
        assert np.array_equal(arrays["x"], data)

        # Check metadata
        assert metadata["horizon"] == 24
        assert metadata["output_type"] == "quantiles"
        assert metadata["quantiles"] == [0.1, 0.5, 0.9]

    def test_to_arrays_and_metadata_without_quantiles(self):
        """Test conversion when quantiles not specified."""
        data = np.array([[1.0, 2.0]])
        request = Chronos2ForecastRequest(
            x=data,
            horizon=10,
            output_type="point",
        )
        arrays, metadata = request.to_arrays_and_metadata()

        assert metadata["output_type"] == "point"
        assert "quantiles" not in metadata


class TestTiRexForecastRequest:
    """Tests for TiRexForecastRequest model."""

    def test_model_name_is_tirex(self):
        """Model name should be TIREX."""
        request = TiRexForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
        )
        assert request.model_name == ModelName.TIREX

    def test_default_values(self):
        """Test default parameter values."""
        request = TiRexForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
        )
        assert request.model_version == "1"
        assert request.compression == "zstd"
        assert request.output_type == "point"

    def test_custom_output_type(self):
        """Test custom output type."""
        request = TiRexForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
            output_type="quantiles",
        )
        assert request.output_type == "quantiles"

    def test_to_arrays_and_metadata(self):
        """Test conversion to Arrow format."""
        data = np.random.rand(10, 50, 2)
        request = TiRexForecastRequest(
            x=data,
            horizon=12,
            output_type="samples",
        )
        arrays, metadata = request.to_arrays_and_metadata()

        assert "x" in arrays
        assert np.array_equal(arrays["x"], data)
        assert metadata["horizon"] == 12
        assert metadata["output_type"] == "samples"


class TestFlowStateForecastRequest:
    """Tests for FlowStateForecastRequest model."""

    def test_model_name_is_flowstate(self):
        """Model name should be FLOWSTATE."""
        request = FlowStateForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
        )
        assert request.model_name == ModelName.FLOWSTATE

    def test_default_values(self):
        """Test default parameter values."""
        request = FlowStateForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
        )
        assert request.model_version == "1"
        assert request.compression == "zstd"
        assert request.output_type == "point"
        assert request.scale_factor is None
        assert request.prediction_type is None

    def test_custom_scale_factor(self):
        """Test custom scale factor."""
        request = FlowStateForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
            scale_factor=100.0,
        )
        assert request.scale_factor == 100.0

    def test_scale_factor_validation_positive(self):
        """Scale factor must be positive."""
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                scale_factor=0.0,
            )

    def test_scale_factor_validation_negative(self):
        """Scale factor cannot be negative."""
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                scale_factor=-1.0,
            )

    def test_prediction_type_mean_with_point_output(self):
        """prediction_type='mean' requires output_type='point'."""
        request = FlowStateForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
            output_type="point",
            prediction_type="mean",
        )
        assert request.prediction_type == "mean"
        assert request.output_type == "point"

    def test_prediction_type_median_with_point_output(self):
        """prediction_type='median' requires output_type='point'."""
        request = FlowStateForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
            output_type="point",
            prediction_type="median",
        )
        assert request.prediction_type == "median"

    def test_prediction_type_quantile_with_quantiles_output(self):
        """prediction_type='quantile' requires output_type='quantiles'."""
        request = FlowStateForecastRequest(
            x=np.array([[1.0, 2.0]]),
            horizon=10,
            output_type="quantiles",
            prediction_type="quantile",
        )
        assert request.prediction_type == "quantile"
        assert request.output_type == "quantiles"

    def test_validation_mean_requires_point_output(self):
        """prediction_type='mean' incompatible with output_type='quantiles'."""
        with pytest.raises(ValueError, match="prediction_type='mean' requires output_type='point'"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                output_type="quantiles",
                prediction_type="mean",
            )

    def test_validation_median_requires_point_output(self):
        """prediction_type='median' incompatible with output_type='quantiles'."""
        with pytest.raises(ValueError, match="prediction_type='median' requires output_type='point'"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                output_type="quantiles",
                prediction_type="median",
            )

    def test_validation_quantile_requires_quantiles_output(self):
        """prediction_type='quantile' requires output_type='quantiles'."""
        with pytest.raises(ValueError, match="prediction_type='quantile' requires output_type='quantiles'"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                output_type="point",
                prediction_type="quantile",
            )

    def test_validation_quantiles_output_requires_quantile_prediction(self):
        """output_type='quantiles' requires prediction_type='quantile'."""
        with pytest.raises(ValueError, match="output_type='quantiles' requires prediction_type='quantile'"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                output_type="quantiles",
                prediction_type="mean",
            )

    def test_validation_quantiles_output_requires_prediction_type(self):
        """output_type='quantiles' requires prediction_type to be set."""
        with pytest.raises(ValueError, match="output_type='quantiles' requires prediction_type='quantile'"):
            FlowStateForecastRequest(
                x=np.array([[1.0, 2.0]]),
                horizon=10,
                output_type="quantiles",
            )

    def test_to_arrays_and_metadata_with_all_params(self):
        """Test conversion with all FlowState parameters."""
        data = np.random.rand(16, 80, 1)
        request = FlowStateForecastRequest(
            x=data,
            horizon=20,
            scale_factor=50.0,
            prediction_type="mean",
            output_type="point",
        )
        arrays, metadata = request.to_arrays_and_metadata()

        assert "x" in arrays
        assert np.array_equal(arrays["x"], data)
        assert metadata["horizon"] == 20
        assert metadata["output_type"] == "point"
        assert metadata["scale_factor"] == 50.0
        assert metadata["prediction_type"] == "mean"

    def test_to_arrays_and_metadata_without_optional_params(self):
        """Test conversion without optional FlowState parameters."""
        data = np.array([[1.0, 2.0]])
        request = FlowStateForecastRequest(
            x=data,
            horizon=10,
        )
        arrays, metadata = request.to_arrays_and_metadata()

        assert metadata["output_type"] == "point"
        assert "scale_factor" not in metadata
        assert "prediction_type" not in metadata


class TestForecastResponse:
    """Tests for ForecastResponse model."""

    def test_from_arrays_point_only(self):
        """Test creating response with point predictions only."""
        point_data = np.random.rand(32, 24, 1)
        arrays = {"point": point_data}
        metadata = {"model_name": "chronos2", "model_version": "1.0"}

        response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)

        assert response.point is not None
        assert np.array_equal(response.point, point_data)
        assert response.quantiles is None
        assert response.samples is None
        assert response.metadata == metadata

    def test_from_arrays_quantiles_only(self):
        """Test creating response with quantile predictions only."""
        quantiles_data = np.random.rand(32, 24, 3)
        arrays = {"quantiles": quantiles_data}
        metadata = {"model_name": "chronos2", "quantiles": [0.1, 0.5, 0.9]}

        response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)

        assert response.point is None
        assert response.quantiles is not None
        assert np.array_equal(response.quantiles, quantiles_data)
        assert response.samples is None

    def test_from_arrays_samples_only(self):
        """Test creating response with sample predictions only."""
        samples_data = np.random.rand(32, 24, 100)
        arrays = {"samples": samples_data}
        metadata = {"model_name": "tirex"}

        response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)

        assert response.point is None
        assert response.quantiles is None
        assert response.samples is not None
        assert np.array_equal(response.samples, samples_data)

    def test_from_arrays_multiple_outputs(self):
        """Test creating response with multiple output types."""
        point_data = np.random.rand(32, 24, 1)
        quantiles_data = np.random.rand(32, 24, 3)
        arrays = {"point": point_data, "quantiles": quantiles_data}
        metadata = {"model_name": "flowstate"}

        response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)

        assert response.point is not None
        assert response.quantiles is not None
        assert response.samples is None

    def test_from_arrays_validation_requires_output(self):
        """Test that at least one output array is required."""
        arrays = {}  # No output arrays
        metadata = {"model_name": "chronos2"}

        with pytest.raises(ValueError, match="Response missing output arrays"):
            ForecastResponse.from_arrays_and_metadata(arrays, metadata)

    def test_from_arrays_ignores_non_output_arrays(self):
        """Test that non-output arrays in dict are ignored."""
        point_data = np.random.rand(32, 24, 1)
        arrays = {"point": point_data, "x": np.random.rand(32, 100, 1)}
        metadata = {}

        response = ForecastResponse.from_arrays_and_metadata(arrays, metadata)
        assert response.point is not None
        # x is not stored in response

    def test_repr_with_point(self):
        """Test string representation with point predictions."""
        response = ForecastResponse(
            point=np.zeros((32, 24, 1)),
            metadata={"model_name": "chronos2"},
        )
        repr_str = repr(response)

        assert "ForecastResponse" in repr_str
        assert "point.shape=(32, 24, 1)" in repr_str
        assert "metadata=" in repr_str

    def test_repr_with_multiple_outputs(self):
        """Test string representation with multiple outputs."""
        response = ForecastResponse(
            point=np.zeros((32, 24, 1)),
            quantiles=np.zeros((32, 24, 3)),
            samples=np.zeros((32, 24, 100)),
            metadata={},
        )
        repr_str = repr(response)

        assert "point.shape=(32, 24, 1)" in repr_str
        assert "quantiles.shape=(32, 24, 3)" in repr_str
        assert "samples.shape=(32, 24, 100)" in repr_str

    def test_repr_with_no_outputs(self):
        """Test string representation with no outputs."""
        response = ForecastResponse(metadata={})
        repr_str = repr(response)

        assert "ForecastResponse" in repr_str
        assert "outputs=[None]" in repr_str

    def test_default_factory_metadata(self):
        """Test that metadata defaults to empty dict."""
        response = ForecastResponse(point=np.zeros((1, 1, 1)))
        assert response.metadata == {}
        assert isinstance(response.metadata, dict)
