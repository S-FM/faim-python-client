"""Unit tests for faim_sdk.utils module.

Tests Arrow serialization and deserialization utilities.
"""

import numpy as np
import pytest

from faim_sdk.utils import deserialize_from_arrow, serialize_to_arrow


class TestSerializeToArrow:
    """Tests for serialize_to_arrow function."""

    def test_serialize_single_array(self):
        """Test serializing a single numpy array."""
        arrays = {"x": np.array([[1.0, 2.0], [3.0, 4.0]])}
        result = serialize_to_arrow(arrays)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_multiple_arrays(self):
        """Test serializing multiple numpy arrays."""
        arrays = {
            "x": np.array([[1.0, 2.0]]),
            "y": np.array([[3.0, 4.0]]),
            "z": np.array([[5.0, 6.0]]),
        }
        result = serialize_to_arrow(arrays)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_with_metadata(self):
        """Test serializing with metadata."""
        arrays = {"x": np.array([[1.0, 2.0]])}
        metadata = {"horizon": 10, "model": "chronos2"}
        result = serialize_to_arrow(arrays, metadata)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_with_compression_zstd(self):
        """Test serializing with zstd compression."""
        arrays = {"x": np.random.rand(100, 100)}
        result = serialize_to_arrow(arrays, compression="zstd")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_with_compression_lz4(self):
        """Test serializing with lz4 compression."""
        arrays = {"x": np.random.rand(100, 100)}
        result = serialize_to_arrow(arrays, compression="lz4")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_without_compression(self):
        """Test serializing without compression."""
        arrays = {"x": np.array([[1.0, 2.0]])}
        result = serialize_to_arrow(arrays, compression=None)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_different_dtypes(self):
        """Test serializing arrays with different dtypes."""
        arrays = {
            "float32": np.array([[1.0, 2.0]], dtype=np.float32),
            "float64": np.array([[3.0, 4.0]], dtype=np.float64),
            "int32": np.array([[5, 6]], dtype=np.int32),
            "int64": np.array([[7, 8]], dtype=np.int64),
        }
        result = serialize_to_arrow(arrays)

        assert isinstance(result, bytes)

    def test_serialize_different_shapes(self):
        """Test serializing arrays with different shapes."""
        arrays = {
            "scalar": np.array([1.0]),
            "1d": np.array([1.0, 2.0, 3.0]),
            "2d": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "3d": np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
        }
        result = serialize_to_arrow(arrays)

        assert isinstance(result, bytes)

    def test_serialize_large_array(self):
        """Test serializing large arrays."""
        arrays = {"large": np.random.rand(1000, 100, 10)}
        result = serialize_to_arrow(arrays)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_empty_metadata(self):
        """Test serializing with empty metadata dict."""
        arrays = {"x": np.array([[1.0]])}
        result = serialize_to_arrow(arrays, metadata={})

        assert isinstance(result, bytes)

    def test_serialize_none_metadata(self):
        """Test serializing with None metadata."""
        arrays = {"x": np.array([[1.0]])}
        result = serialize_to_arrow(arrays, metadata=None)

        assert isinstance(result, bytes)

    def test_serialize_complex_metadata(self):
        """Test serializing with complex nested metadata."""
        arrays = {"x": np.array([[1.0]])}
        metadata = {
            "horizon": 10,
            "quantiles": [0.1, 0.5, 0.9],
            "config": {"model": "chronos2", "version": "1.0"},
        }
        result = serialize_to_arrow(arrays, metadata)

        assert isinstance(result, bytes)

    def test_serialize_skips_none_arrays(self):
        """Test that None values in arrays dict are skipped."""
        arrays = {"x": np.array([[1.0]]), "y": None, "z": np.array([[2.0]])}
        result = serialize_to_arrow(arrays)

        # Should succeed without error
        assert isinstance(result, bytes)

    def test_serialize_validation_requires_numpy_array(self):
        """Test that non-numpy arrays raise TypeError."""
        arrays = {"x": [[1.0, 2.0], [3.0, 4.0]]}  # Python list

        with pytest.raises(TypeError, match='Array "x" must be numpy.ndarray'):
            serialize_to_arrow(arrays)

    def test_serialize_validation_rejects_list(self):
        """Test that lists raise TypeError."""
        arrays = {"data": [1, 2, 3, 4, 5]}

        with pytest.raises(TypeError, match="must be numpy.ndarray"):
            serialize_to_arrow(arrays)

    def test_serialize_non_native_endianness(self):
        """Test handling of non-native endianness arrays."""
        # Create array with non-native byte order
        arr = np.array([[1.0, 2.0]], dtype=">f8")  # Big-endian
        arrays = {"x": arr}

        # Should handle conversion automatically
        result = serialize_to_arrow(arrays)
        assert isinstance(result, bytes)

    def test_serialize_non_contiguous_array(self):
        """Test handling of non-contiguous arrays."""
        # Create non-contiguous array via transpose
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T
        assert not arr.flags.c_contiguous

        arrays = {"x": arr}
        result = serialize_to_arrow(arrays)

        assert isinstance(result, bytes)

    def test_serialize_deterministic_order(self):
        """Test that serialization produces deterministic ordering."""
        arrays1 = {"z": np.array([3.0]), "a": np.array([1.0]), "m": np.array([2.0])}
        arrays2 = {"a": np.array([1.0]), "m": np.array([2.0]), "z": np.array([3.0])}

        result1 = serialize_to_arrow(arrays1, compression=None)
        result2 = serialize_to_arrow(arrays2, compression=None)

        # Should produce identical bytes due to sorted keys
        assert result1 == result2


class TestDeserializeFromArrow:
    """Tests for deserialize_from_arrow function."""

    def test_deserialize_single_array(self):
        """Test deserializing a single array."""
        original = {"x": np.array([[1.0, 2.0], [3.0, 4.0]])}
        serialized = serialize_to_arrow(original)

        arrays, metadata = deserialize_from_arrow(serialized)

        assert "x" in arrays
        assert np.array_equal(arrays["x"], original["x"])

    def test_deserialize_multiple_arrays(self):
        """Test deserializing multiple arrays."""
        original = {
            "x": np.array([[1.0, 2.0]]),
            "y": np.array([[3.0, 4.0]]),
            "z": np.array([[5.0, 6.0]]),
        }
        serialized = serialize_to_arrow(original)

        arrays, metadata = deserialize_from_arrow(serialized)

        assert set(arrays.keys()) == {"x", "y", "z"}
        assert np.array_equal(arrays["x"], original["x"])
        assert np.array_equal(arrays["y"], original["y"])
        assert np.array_equal(arrays["z"], original["z"])

    def test_deserialize_with_metadata(self):
        """Test deserializing with metadata."""
        original_arrays = {"x": np.array([[1.0]])}
        original_metadata = {"horizon": 10, "model": "chronos2"}
        serialized = serialize_to_arrow(original_arrays, original_metadata)

        arrays, metadata = deserialize_from_arrow(serialized)

        assert metadata == original_metadata
        assert metadata["horizon"] == 10
        assert metadata["model"] == "chronos2"

    def test_deserialize_preserves_dtype(self):
        """Test that deserialization preserves dtypes."""
        original = {
            "float32": np.array([[1.0, 2.0]], dtype=np.float32),
            "float64": np.array([[3.0, 4.0]], dtype=np.float64),
            "int32": np.array([[5, 6]], dtype=np.int32),
            "int64": np.array([[7, 8]], dtype=np.int64),
        }
        serialized = serialize_to_arrow(original)

        arrays, _ = deserialize_from_arrow(serialized)

        assert arrays["float32"].dtype == np.float32
        assert arrays["float64"].dtype == np.float64
        assert arrays["int32"].dtype == np.int32
        assert arrays["int64"].dtype == np.int64

    def test_deserialize_preserves_shape(self):
        """Test that deserialization preserves shapes."""
        original = {
            "scalar": np.array([1.0]),
            "1d": np.array([1.0, 2.0, 3.0]),
            "2d": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "3d": np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
        }
        serialized = serialize_to_arrow(original)

        arrays, _ = deserialize_from_arrow(serialized)

        assert arrays["scalar"].shape == (1,)
        assert arrays["1d"].shape == (3,)
        assert arrays["2d"].shape == (2, 2)
        assert arrays["3d"].shape == (2, 1, 2)

    def test_deserialize_large_array(self):
        """Test deserializing large arrays."""
        original = {"large": np.random.rand(1000, 100, 10)}
        serialized = serialize_to_arrow(original)

        arrays, _ = deserialize_from_arrow(serialized)

        assert "large" in arrays
        assert arrays["large"].shape == (1000, 100, 10)
        assert np.array_equal(arrays["large"], original["large"])

    def test_deserialize_empty_metadata(self):
        """Test deserializing with empty metadata."""
        original = {"x": np.array([[1.0]])}
        serialized = serialize_to_arrow(original, metadata={})

        arrays, metadata = deserialize_from_arrow(serialized)

        assert metadata == {}

    def test_deserialize_none_metadata(self):
        """Test deserializing with None metadata."""
        original = {"x": np.array([[1.0]])}
        serialized = serialize_to_arrow(original, metadata=None)

        arrays, metadata = deserialize_from_arrow(serialized)

        assert metadata == {}

    def test_deserialize_complex_metadata(self):
        """Test deserializing with complex nested metadata."""
        original_metadata = {
            "horizon": 10,
            "quantiles": [0.1, 0.5, 0.9],
            "config": {"model": "chronos2", "version": "1.0"},
        }
        original_arrays = {"x": np.array([[1.0]])}
        serialized = serialize_to_arrow(original_arrays, original_metadata)

        arrays, metadata = deserialize_from_arrow(serialized)

        assert metadata == original_metadata
        assert metadata["quantiles"] == [0.1, 0.5, 0.9]
        assert metadata["config"]["model"] == "chronos2"


class TestRoundTrip:
    """Tests for serialize -> deserialize round-trip consistency."""

    def test_roundtrip_simple_array(self):
        """Test round-trip for simple array."""
        original = {"x": np.array([[1.0, 2.0], [3.0, 4.0]])}
        serialized = serialize_to_arrow(original)
        arrays, _ = deserialize_from_arrow(serialized)

        assert np.array_equal(arrays["x"], original["x"])

    def test_roundtrip_with_metadata(self):
        """Test round-trip preserves metadata."""
        original_arrays = {"x": np.array([[1.0]])}
        original_metadata = {"horizon": 24, "output_type": "quantiles"}

        serialized = serialize_to_arrow(original_arrays, original_metadata)
        arrays, metadata = deserialize_from_arrow(serialized)

        assert metadata == original_metadata

    def test_roundtrip_multiple_arrays(self):
        """Test round-trip for multiple arrays."""
        original = {
            "x": np.random.rand(32, 100, 1),
            "point": np.random.rand(32, 24, 1),
            "quantiles": np.random.rand(32, 24, 3),
        }
        serialized = serialize_to_arrow(original)
        arrays, _ = deserialize_from_arrow(serialized)

        for key in original:
            assert key in arrays
            assert np.array_equal(arrays[key], original[key])
            assert arrays[key].shape == original[key].shape
            assert arrays[key].dtype == original[key].dtype

    def test_roundtrip_with_zstd_compression(self):
        """Test round-trip with zstd compression."""
        original = {"x": np.random.rand(100, 50)}
        serialized = serialize_to_arrow(original, compression="zstd")
        arrays, _ = deserialize_from_arrow(serialized)

        assert np.allclose(arrays["x"], original["x"])

    def test_roundtrip_with_lz4_compression(self):
        """Test round-trip with lz4 compression."""
        original = {"x": np.random.rand(100, 50)}
        serialized = serialize_to_arrow(original, compression="lz4")
        arrays, _ = deserialize_from_arrow(serialized)

        assert np.allclose(arrays["x"], original["x"])

    def test_roundtrip_different_dtypes(self):
        """Test round-trip preserves all dtypes."""
        original = {
            "float32": np.array([1.5, 2.5], dtype=np.float32),
            "float64": np.array([3.5, 4.5], dtype=np.float64),
            "int32": np.array([5, 6], dtype=np.int32),
            "int64": np.array([7, 8], dtype=np.int64),
        }
        serialized = serialize_to_arrow(original, compression=None)
        arrays, _ = deserialize_from_arrow(serialized)

        for key in original:
            assert arrays[key].dtype == original[key].dtype
            assert np.array_equal(arrays[key], original[key])

    def test_roundtrip_different_shapes(self):
        """Test round-trip preserves all shapes."""
        original = {
            "1d": np.array([1.0, 2.0, 3.0]),
            "2d": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "3d": np.random.rand(4, 5, 6),
            "4d": np.random.rand(2, 3, 4, 5),
        }
        serialized = serialize_to_arrow(original)
        arrays, _ = deserialize_from_arrow(serialized)

        for key in original:
            assert arrays[key].shape == original[key].shape
            assert np.allclose(arrays[key], original[key])

    def test_roundtrip_exact_values(self):
        """Test round-trip preserves exact floating point values."""
        original = {"data": np.array([1.23456789, 9.87654321, -0.123456, 0.0])}
        serialized = serialize_to_arrow(original, compression=None)
        arrays, _ = deserialize_from_arrow(serialized)

        # Should be exactly equal (no floating point error)
        assert np.array_equal(arrays["data"], original["data"])

    def test_roundtrip_negative_values(self):
        """Test round-trip with negative values."""
        original = {"data": np.array([[-1.0, -2.0], [-3.0, -4.0]])}
        serialized = serialize_to_arrow(original)
        arrays, _ = deserialize_from_arrow(serialized)

        assert np.array_equal(arrays["data"], original["data"])

    def test_roundtrip_zero_values(self):
        """Test round-trip with zero values."""
        original = {"data": np.zeros((10, 10))}
        serialized = serialize_to_arrow(original)
        arrays, _ = deserialize_from_arrow(serialized)

        assert np.array_equal(arrays["data"], original["data"])

    def test_roundtrip_special_floats(self):
        """Test round-trip with special float values."""
        original = {"data": np.array([np.inf, -np.inf, 0.0, -0.0, 1e-300, 1e300])}
        serialized = serialize_to_arrow(original)
        arrays, _ = deserialize_from_arrow(serialized)

        assert np.array_equal(arrays["data"], original["data"])

    def test_roundtrip_realistic_forecast_request(self):
        """Test round-trip for realistic forecast request data."""
        original_arrays = {
            "x": np.random.randn(32, 100, 1).astype(np.float32),
        }
        original_metadata = {
            "horizon": 24,
            "model_version": "1.0",
            "output_type": "quantiles",
            "quantiles": [0.1, 0.5, 0.9],
        }

        serialized = serialize_to_arrow(original_arrays, original_metadata, compression="zstd")
        arrays, metadata = deserialize_from_arrow(serialized)

        assert np.allclose(arrays["x"], original_arrays["x"])
        assert arrays["x"].shape == (32, 100, 1)
        assert arrays["x"].dtype == np.float32
        assert metadata == original_metadata

    def test_roundtrip_realistic_forecast_response(self):
        """Test round-trip for realistic forecast response data."""
        original_arrays = {
            "point": np.random.randn(32, 24, 1).astype(np.float32),
            "quantiles": np.random.randn(32, 24, 3).astype(np.float32),
        }
        original_metadata = {
            "model_name": "chronos2",
            "model_version": "1.0",
            "inference_time_ms": 123,
        }

        serialized = serialize_to_arrow(original_arrays, original_metadata, compression="zstd")
        arrays, metadata = deserialize_from_arrow(serialized)

        assert set(arrays.keys()) == {"point", "quantiles"}
        assert np.allclose(arrays["point"], original_arrays["point"])
        assert np.allclose(arrays["quantiles"], original_arrays["quantiles"])
        assert metadata == original_metadata
