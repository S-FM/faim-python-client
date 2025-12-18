"""Utility functions for Arrow serialization and data conversion.

Provides specialized serialization for:
- Time-series data: serialize_to_arrow() - optimized for single input arrays using RecordBatch
- Tabular data: serialize_to_arrow_tabular() - supports multiple arrays with different row counts using Arrow Table ("tensor table" format)
"""

import json
from typing import Any

import numpy as np
import pyarrow as pa


def _validate_and_prepare_array(name: str, arr: np.ndarray) -> np.ndarray:
    """Validate and prepare numpy array for Arrow serialization.

    Internal helper function for both time-series and tabular serialization.
    Performs common preprocessing: type checking, endianness fix, and contiguity.

    Args:
        name: Array name (for error messages)
        arr: Numpy array to prepare

    Returns:
        Prepared numpy array (may be a view or copy depending on original state)

    Raises:
        TypeError: If array is not a numpy ndarray
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f'Array "{name}" must be numpy.ndarray, got {type(arr).__name__}')

    # Ensure native endianness for Arrow compatibility
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

    # Ensure C-contiguous layout for performance and Arrow compatibility
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    return arr


def _prepare_arrow_field_and_array(name: str, arr: np.ndarray) -> tuple[pa.Field, pa.Array]:
    """Convert numpy array to Arrow field and array with metadata.

    Internal helper function shared by both time-series and tabular serialization.
    Handles dtype conversion, endianness, and shape metadata storage.

    Args:
        name: Field name in schema
        arr: Numpy array to convert

    Returns:
        Tuple of (Arrow Field, Arrow Array)

    Raises:
        TypeError: If array is not a numpy ndarray
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f'Array "{name}" must be numpy.ndarray, got {type(arr).__name__}')

    # Ensure native endianness for Arrow compatibility
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)

    # Ensure C-contiguous layout for performance
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    # Store original shape and dtype in field metadata for reconstruction
    field_meta = {
        b"shape": json.dumps(list(arr.shape)).encode("utf-8"),
        b"dtype": str(arr.dtype).encode("utf-8"),
    }

    # Create Arrow field with correct type
    pa_field = pa.field(name, pa.from_numpy_dtype(arr.dtype), metadata=field_meta)

    # Convert to Arrow array (zero-copy when possible)
    flattened = arr.ravel()
    arrow_array = pa.array(flattened, type=pa_field.type, from_pandas=True)

    return pa_field, arrow_array


def serialize_to_arrow(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    compression: str | None = "zstd",
) -> bytes:
    """Serialize numpy arrays to Arrow IPC stream format (time-series optimized).

    Optimized for time-series data with a single input array. Uses RecordBatch
    for maximum efficiency. For tabular data with multiple arrays of different
    row counts, use serialize_to_arrow_tabular() instead.

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
        >>> arrays = {"x": np.array([[[1, 2], [3, 4]]])}  # (batch=1, seq=2, features=2)
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

        field, col = _prepare_arrow_field_and_array(name, arr)
        fields.append(field)
        cols.append(col)

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


def _deserialize_arrow_table(table: pa.Table) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract arrays and metadata from Arrow Table.

    Internal helper function shared by both time-series and tabular deserialization.
    Reconstructs numpy arrays from flattened Arrow columns and restores original shapes.

    Args:
        table: Arrow Table to deserialize

    Returns:
        Tuple of (arrays dict, metadata dict)
    """
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


def deserialize_from_arrow(arrow_bytes: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Deserialize Arrow IPC stream to numpy arrays and metadata (time-series optimized).

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
    table = reader.read_all()
    return _deserialize_arrow_table(table)


def serialize_to_arrow_tabular(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    compression: str | None = "zstd",
) -> bytes:
    """Serialize numpy arrays to Arrow IPC stream format (tabular optimized).

    Uses "tensor table" format: Each array becomes one row in an Arrow Table.
    Supports arrays with different row counts and shapes.

    Schema:
    - array_name (string): Name of the array
    - data (binary): Serialized binary data (from array.tobytes())
    - shape (string): JSON-encoded shape tuple
    - dtype (string): NumPy dtype name

    For time-series data with a single input array, use serialize_to_arrow() instead
    for better performance.

    Args:
        arrays: Dictionary mapping names to numpy arrays (can have different shapes)
        metadata: Optional user metadata to include in schema
        compression: Compression algorithm ('zstd', 'lz4', None). Default: 'zstd'

    Returns:
        Serialized Arrow IPC stream as bytes

    Raises:
        TypeError: If array value is not a numpy ndarray
        ValueError: If serialization fails

    Example:
        >>> X_train = np.random.randn(100, 10).astype(np.float32)
        >>> y_train = np.random.randint(0, 2, 100).astype(np.float32)
        >>> X_test = np.random.randn(20, 10).astype(np.float32)  # Different row count!
        >>> arrays = {"X_train": X_train, "y_train": y_train, "X_test": X_test}
        >>> arrow_bytes = serialize_to_arrow_tabular(arrays, metadata={"task": "classification"})
    """
    # Build tensor table rows: one row per array
    array_names = []
    data_blobs = []
    shape_strs = []
    dtype_strs = []

    # Deterministic order for reproducibility
    for name in sorted(arrays.keys()):
        arr = arrays[name]

        # Skip None values (optional arrays)
        if arr is None:
            continue

        # Validate and prepare array (common preprocessing)
        arr = _validate_and_prepare_array(name, arr)

        # Store array metadata and binary data
        array_names.append(name)
        data_blobs.append(arr.tobytes())
        shape_strs.append(json.dumps(list(arr.shape)))
        dtype_strs.append(str(arr.dtype))

    # Create Arrow Table with one row per array (tensor table format)
    tensor_table_data = {
        "array_name": array_names,
        "data": data_blobs,
        "shape": shape_strs,
        "dtype": dtype_strs,
    }

    try:
        table = pa.Table.from_pydict(tensor_table_data)
    except Exception as e:
        raise ValueError(f"Failed to create Arrow tensor table: {e}") from e

    # Embed user metadata in schema
    schema_meta = {b"user_meta": json.dumps(metadata or {}).encode("utf-8")}
    table = table.replace_schema_metadata(schema_meta)

    # Serialize IPC stream with optional compression
    sink = pa.BufferOutputStream()
    write_options = pa.ipc.IpcWriteOptions(compression=compression)

    try:
        with pa.ipc.new_stream(sink, table.schema, options=write_options) as writer:
            # Write all rows as a single batch
            for batch in table.to_batches():
                writer.write_batch(batch)
    except Exception as e:
        raise ValueError(f"Failed to serialize Arrow tensor table: {e}") from e

    return sink.getvalue().to_pybytes()


def deserialize_from_arrow_tabular(arrow_bytes: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Deserialize Arrow IPC stream to numpy arrays and metadata (tabular optimized).

    Reconstructs tensor table format: reads rows and reconstructs each array from
    binary data, shape, and dtype metadata.

    Args:
        arrow_bytes: Arrow IPC stream bytes

    Returns:
        Tuple of (arrays dict, metadata dict)

    Raises:
        ValueError: If deserialization fails or stream is invalid

    Example:
        >>> arrays, metadata = deserialize_from_arrow_tabular(arrow_bytes)
        >>> print(arrays["X_train"].shape)  # (100, 10)
        >>> print(arrays["X_test"].shape)   # (20, 10)  - different row count!
    """
    try:
        reader = pa.ipc.open_stream(pa.py_buffer(arrow_bytes))
        table = reader.read_all()
    except Exception as e:
        raise ValueError(f"Failed to read Arrow stream: {e}") from e

    # Extract metadata from schema
    result_metadata = {}
    if table.schema.metadata and b"user_meta" in table.schema.metadata:
        try:
            result_metadata = json.loads(table.schema.metadata[b"user_meta"].decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse user metadata: {e}") from e

    # Check if this is a tensor table (has array_name, data, shape, dtype columns)
    schema_names = {field.name for field in table.schema}
    is_tensor_table = schema_names >= {"array_name", "data", "shape", "dtype"}

    if not is_tensor_table:
        raise ValueError(
            f"Tensor table must have columns: array_name, data, shape, dtype. Got columns: {sorted(schema_names)}"
        )

    # Extract tensor table columns
    try:
        array_names_col = table["array_name"].to_pylist()
        data_col = table["data"].to_pylist()
        shape_col = table["shape"].to_pylist()
        dtype_col = table["dtype"].to_pylist()
    except KeyError as e:
        raise ValueError(f"Missing required column in tensor table: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to extract tensor table columns: {e}") from e

    # Reconstruct arrays from binary data
    result_arrays = {}
    for array_name, binary_data, shape_json, dtype_str in zip(array_names_col, data_col, shape_col, dtype_col):
        try:
            # Parse shape and dtype
            shape = json.loads(shape_json)
            dtype = np.dtype(dtype_str)

            # Reconstruct array from binary data
            arr = np.frombuffer(binary_data, dtype=dtype)
            arr = arr.reshape(shape)

            result_arrays[array_name] = arr
        except Exception as e:
            raise ValueError(f"Failed to reconstruct array '{array_name}': {e}") from e

    return result_arrays, result_metadata
