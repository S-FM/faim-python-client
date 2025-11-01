"""Utility functions for Arrow serialization and data conversion."""

import json
from typing import Any

import numpy as np
import pyarrow as pa


def serialize_to_arrow(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    compression: str | None = "zstd",
) -> bytes:
    """Serialize numpy arrays to Arrow IPC stream format.

    Args:
        arrays: Dictionary mapping names to numpy arrays
        metadata: Optional metadata to include in schema
        compression: Compression algorithm ('zstd', 'lz4', None). Default: 'zstd'

    Returns:
        Serialized Arrow IPC stream as bytes

    Raises:
        TypeError: If array is not a numpy ndarray
        ValueError: If serialization fails

    Example:
        >>> arrays = {"x": np.array([[1, 2], [3, 4]])}
        >>> metadata = {"horizon": 10}
        >>> arrow_bytes = serialize_to_arrow(arrays, metadata)
    """
    fields, cols = [], []

    # Deterministic order for reproducibility
    for name in sorted(arrays.keys()):
        arr = arrays[name]

        # Skip None values (optional arrays)
        if arr is None:
            continue

        if not isinstance(arr, np.ndarray):
            raise TypeError(f'Array "{name}" must be numpy.ndarray, got {type(arr).__name__}')

        # Ensure native endianness for Arrow compatibility
        if not arr.dtype.isnative:
            arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

        # Ensure C-contiguous layout for performance
        # PERFORMANCE FIX: Only copy if necessary
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        # Store original shape and dtype in field metadata for reconstruction
        field_meta = {
            b"shape": json.dumps(list(arr.shape)).encode("utf-8"),
            b"dtype": str(arr.dtype).encode("utf-8"),
        }

        # Create Arrow field with correct type
        pa_field = pa.field(name, pa.from_numpy_dtype(arr.dtype), metadata=field_meta)
        fields.append(pa_field)

        # PERFORMANCE OPTIMIZATION: Zero-copy conversion from numpy to Arrow
        # ravel() returns view for C-contiguous arrays (no copy)
        flattened = arr.ravel()

        # pa.array() with from_pandas flag attempts zero-copy when possible
        # For primitive types with native endianness, Arrow can use numpy's buffer directly
        arrow_array = pa.array(flattened, type=pa_field.type, from_pandas=True)
        cols.append(arrow_array)

    # Embed user metadata in schema
    schema_meta = {b"user_meta": json.dumps(metadata or {}).encode("utf-8")}
    schema = pa.schema(fields, metadata=schema_meta)
    batch = pa.record_batch(cols, schema=schema)

    # Serialize with optional compression
    sink = pa.BufferOutputStream()
    write_options = pa.ipc.IpcWriteOptions(compression=compression)

    with pa.ipc.new_stream(sink, schema, options=write_options) as writer:
        writer.write_batch(batch)

    return sink.getvalue().to_pybytes()


def deserialize_from_arrow(arrow_bytes: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Deserialize Arrow IPC stream to numpy arrays and metadata.

    Optimized for batch inference: reads all batches efficiently using Arrow Table
    for zero-copy conversion to numpy when possible.

    Args:
        arrow_bytes: Arrow IPC stream bytes

    Returns:
        Tuple of (arrays dict, metadata dict)

    Raises:
        ValueError: If deserialization fails or stream is invalid

    Example:
        >>> # Single batch
        >>> arrays, metadata = deserialize_from_arrow(arrow_bytes)
        >>> print(arrays["predictions"].shape)
        (32, 10)  # batch_size=32, horizon=10
    """
    reader = pa.ipc.open_stream(pa.py_buffer(arrow_bytes))

    # PERFORMANCE OPTIMIZATION: Use read_all() instead of manual batch loop
    # read_all() returns a Table that efficiently handles:
    # - Single batch: Zero-copy access to underlying batch
    # - Multiple batches: Efficient chunked column representation without immediate concat
    # - Arrow's internal optimizations for column access
    table = reader.read_all()

    # Extract arrays with shape reconstruction
    result_arrays = {}
    for i, field in enumerate(table.schema):
        # PERFORMANCE: to_numpy() on chunked columns is optimized:
        # - Single chunk: zero-copy view
        # - Multiple chunks: efficient concatenation with pre-allocated buffer
        col_chunked = table.column(i)
        arr_np = col_chunked.to_numpy(zero_copy_only=False)  # Allow copy for correctness

        # Reconstruct original shape from field metadata
        if field.metadata and b"shape" in field.metadata:
            shape = json.loads(field.metadata[b"shape"].decode("utf-8"))
            dtype = np.dtype(field.metadata[b"dtype"].decode("utf-8"))

            # PERFORMANCE: Use copy=False to avoid unnecessary copies
            # Only copies if dtype conversion requires it
            result_arrays[field.name] = arr_np.astype(dtype, copy=False).reshape(shape)
        else:
            result_arrays[field.name] = arr_np

    # Extract user metadata from schema
    result_metadata = {}
    if table.schema.metadata and b"user_meta" in table.schema.metadata:
        result_metadata = json.loads(table.schema.metadata[b"user_meta"].decode("utf-8"))

    return result_arrays, result_metadata
