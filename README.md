# FAIM SDK

[![PyPI version](https://badge.fury.io/py/faim-sdk.svg)](https://badge.fury.io/py/faim-sdk)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Production-ready Python SDK for FAIM (Foundation AI Models) - a high-performance time-series forecasting platform powered by foundation models.

## Features

- **🚀 Multiple Foundation Models**: FlowState, Amazon Chronos 2.0, TiRex
- **🔒 Type-Safe API**: Full type hints with Pydantic validation
- **⚡ High Performance**: Optimized Apache Arrow serialization with zero-copy operations
- **🎯 Probabilistic & Deterministic**: Point forecasts, quantiles, and samples
- **🔄 Async Support**: Built-in async/await support for concurrent requests
- **📊 Rich Error Handling**: Machine-readable error codes with detailed diagnostics
- **🧪 Battle-Tested**: Production-ready with comprehensive error handling

## Installation

```bash
pip install faim-sdk
```

## Quick Start

```python
import numpy as np
from faim_sdk import ForecastClient, Chronos2ForecastRequest
from faim_client.models import ModelName

# Initialize client with your API endpoint
client = ForecastClient(
    base_url="https://api.faim.example.com",
    api_key="your-api-key",  # Optional: for authenticated endpoints
    timeout=120.0
)

# Prepare your time-series data
# Shape: (batch_size, sequence_length, features)
data = np.random.randn(32, 100, 1).astype(np.float32)

# Create probabilistic forecast request
request = Chronos2ForecastRequest(
    x=data,
    horizon=24,  # Forecast 24 steps ahead
    output_type="quantiles",
    quantiles=[0.1, 0.5, 0.9]  # 10th, 50th (median), 90th percentiles
)

# Generate forecast
response = client.forecast(ModelName.CHRONOS2, request)

# Access predictions
print(response.quantiles.shape)  # (32, 24, 3)
print(response.metadata)  # Model version, inference time, etc.
```

## Available Models

### FlowState

Optimized for **deterministic point forecasts** with optional scaling and normalization.

```python
from faim_sdk import FlowStateForecastRequest

request = FlowStateForecastRequest(
    x=data,
    horizon=24,
    model_version="latest",
    output_type="point",
    scale_factor=1.0,  # Optional: normalization factor
    prediction_type="mean"  # Options: "mean", "median"
)

response = client.forecast(ModelName.FLOWSTATE, request)
print(response.point.shape)  # (batch_size, 24, features)
```

### Chronos 2.0

Amazon's **large language model for time series** - ideal for probabilistic forecasting with quantiles.

```python
from faim_sdk import Chronos2ForecastRequest

# Quantile-based probabilistic forecast
request = Chronos2ForecastRequest(
    x=data,
    horizon=24,
    output_type="quantiles",
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]  # Full distribution
)

response = client.forecast(ModelName.CHRONOS2, request)
print(response.quantiles.shape)  # (batch_size, 24, 5)
```

### TiRex

**Transformer-based forecasting** model for efficient and accurate predictions.

```python
from faim_sdk import TiRexForecastRequest

request = TiRexForecastRequest(
    x=data,
    horizon=24,
    output_type="point"
)

response = client.forecast(ModelName.TIREX, request)
print(response.point.shape)  # (batch_size, 24, features)
```

## Response Format

All forecasts return a `ForecastResponse` object with predictions and metadata:

```python
response = client.forecast(ModelName.CHRONOS2, request)

# Access predictions based on output_type
if response.point is not None:
    predictions = response.point  # Shape: (batch_size, horizon, features)

if response.quantiles is not None:
    quantiles = response.quantiles  # Shape: (batch_size, horizon, num_quantiles)
    # Lower quantiles for uncertainty bounds
    lower_bound = quantiles[:, :, 0]  # 10th percentile
    median = quantiles[:, :, 1]       # 50th percentile (median)
    upper_bound = quantiles[:, :, 2]  # 90th percentile

if response.samples is not None:
    samples = response.samples  # Shape: (batch_size, horizon, num_samples)

# Access metadata
print(response.metadata)
# {'model_name': 'chronos2', 'model_version': '1.0', 'inference_time_ms': 123}
```

## Error Handling

The SDK provides **machine-readable error codes** for robust error handling:

```python
from faim_sdk import (
    ForecastClient,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ErrorCode
)

try:
    response = client.forecast(ModelName.CHRONOS2, request)

except AuthenticationError as e:
    # Handle authentication failures (401, 403)
    print(f"Authentication failed: {e.message}")
    print(f"Request ID: {e.error_response.request_id}")

except ValidationError as e:
    # Handle invalid request parameters (422)
    if e.error_code == ErrorCode.INVALID_SHAPE:
        print(f"Shape error: {e.error_response.detail}")
        # Fix shape and retry
    elif e.error_code == ErrorCode.MISSING_REQUIRED_FIELD:
        print(f"Missing field: {e.error_response.detail}")

except RateLimitError as e:
    # Handle rate limiting (429)
    print("Rate limit exceeded - implementing exponential backoff")
    retry_after = e.error_response.metadata.get('retry_after', 60)
    time.sleep(retry_after)

except ModelNotFoundError as e:
    # Handle model/version not found (404)
    print(f"Model not found: {e.message}")
```

### Exception Hierarchy

```
FAIMError (base)
├── APIError
│   ├── AuthenticationError (401, 403)
│   ├── InsufficientFundsError (402)
│   ├── ModelNotFoundError (404)
│   ├── PayloadTooLargeError (413)
│   ├── ValidationError (422)
│   ├── RateLimitError (429)
│   ├── InternalServerError (500)
│   └── ServiceUnavailableError (503, 504)
├── NetworkError
├── SerializationError
├── TimeoutError
└── ConfigurationError
```

## Async Support

The SDK supports async operations for concurrent requests:

```python
import asyncio
from faim_sdk import ForecastClient, Chronos2ForecastRequest
from faim_client.models import ModelName

async def forecast_multiple_series():
    client = ForecastClient(
        base_url="https://api.faim.example.com",
        api_key="your-api-key"
    )

    # Create multiple requests
    requests = [
        Chronos2ForecastRequest(x=data1, horizon=24),
        Chronos2ForecastRequest(x=data2, horizon=24),
        Chronos2ForecastRequest(x=data3, horizon=24),
    ]

    # Execute concurrently
    async with client:
        tasks = [
            client.forecast_async(ModelName.CHRONOS2, req)
            for req in requests
        ]
        responses = await asyncio.gather(*tasks)

    return responses

# Run async forecasts
responses = asyncio.run(forecast_multiple_series())
```

## Configuration

### Client Options

```python
from faim_sdk import ForecastClient

# Basic configuration
client = ForecastClient(
    base_url="https://api.faim.example.com",
    timeout=120.0,  # Request timeout in seconds (default: 120)
    verify_ssl=True,  # SSL certificate verification (default: True)
)

# With API key authentication
client = ForecastClient(
    base_url="https://api.faim.example.com",
    api_key="your-secret-api-key",
    timeout=120.0
)

# Advanced configuration with custom httpx settings
import httpx

client = ForecastClient(
    base_url="https://api.faim.example.com",
    api_key="your-api-key",
    timeout=120.0,
    limits=httpx.Limits(max_connections=10),  # Connection pooling
    headers={"X-Custom-Header": "value"}  # Custom headers
)
```

### Request Options

```python
# Compression options for large payloads
request = Chronos2ForecastRequest(
    x=data,
    horizon=24,
    compression="zstd"  # Options: "zstd", "lz4", None (default: "zstd")
)

# Model version pinning
request = FlowStateForecastRequest(
    x=data,
    horizon=24,
    model_version="1.2.3"  # Pin to specific version (default: "latest")
)
```

## Context Managers

Use context managers for automatic resource cleanup:

```python
# Sync context manager
with ForecastClient(base_url="https://api.faim.example.com") as client:
    response = client.forecast(ModelName.CHRONOS2, request)
    print(response.quantiles)
# Client automatically closed

# Async context manager
async with ForecastClient(base_url="https://api.faim.example.com") as client:
    response = await client.forecast_async(ModelName.CHRONOS2, request)
    print(response.quantiles)
# Client automatically closed
```

## Examples

See the `examples/` directory for complete Jupyter notebook examples:

- **`flowstate_simple_example.ipynb`** - Point forecasting with FlowState on AirPassengers dataset
- **`chronos2_probabilistic.ipynb`** - Probabilistic forecasting with quantiles (coming soon)
- **`batch_processing.ipynb`** - Efficient batch processing patterns (coming soon)

## Requirements

- Python >= 3.11
- numpy >= 1.26.0
- pyarrow >= 11.0.0
- httpx >= 0.23.0
- pydantic >= 2.0.0

## Performance Tips

1. **Batch Processing**: Process multiple time series in a single request for optimal throughput
   ```python
   # Good: Single request with 32 series
   data = np.random.randn(32, 100, 1)

   # Less efficient: 32 separate requests
   # for series in data: client.forecast(...)
   ```

2. **Compression**: Use `compression="zstd"` for large payloads (default, recommended)

3. **Async for Concurrent Requests**: Use `forecast_async()` with `asyncio.gather()` for parallel processing

4. **Connection Pooling**: Reuse client instances across requests instead of creating new ones

## Support

- **Documentation**: [docs.faim.example.com](https://docs.faim.example.com)
- **Issues**: [GitHub Issues](https://github.com/your-org/faim-sdk/issues)
- **Email**: support@faim.example.com

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Citation

If you use FAIM in your research, please cite:

```bibtex
@software{faim_sdk,
  title = {FAIM SDK: Foundation AI Models for Time Series Forecasting},
  author = {FAIM Team},
  year = {2024},
  url = {https://github.com/your-org/faim-sdk}
}
```
