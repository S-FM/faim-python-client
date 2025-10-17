# FAIM SDK

Python SDK for FAIM time-series forecasting with foundation AI models.

## Installation

```bash
pip install faim-sdk
```

## Quick Start

```python
import numpy as np
from faim_sdk import ForecastClient, FlowStateForecastRequest
from faim_client.models import ModelName

# Initialize client
client = ForecastClient(
    base_url="http://localhost:8003",
    timeout=60.0
)

# Prepare your time-series data
# Shape: (batch_size, sequence_length, features)
data = np.random.randn(1, 100, 1).astype(np.float32)

# Create forecast request
request = FlowStateForecastRequest(
    x=data,
    horizon=10,
    model_version="1"
)

# Generate forecast
response = client.forecast(ModelName.FLOWSTATE, request)

# Access predictions
print(response.point)  # Shape: (batch_size, horizon, features)
print(response.metadata)  # Model metadata
```

## Models

### FlowState

Point forecasting model optimized for deterministic predictions.

```python
from faim_sdk import FlowStateForecastRequest

request = FlowStateForecastRequest(
    x=data,
    horizon=10,
    model_version="1",
    output_type="point",  # Options: "point", "quantiles", "samples"
    scale_factor=1.0,  # Optional normalization
    prediction_type="mean"  # Options: "mean", "median", "quantile"
)
```

### ToTo

Probabilistic forecasting model with quantile and sample-based predictions.

```python
from faim_sdk import ToToForecastRequest

# Quantile predictions
request = ToToForecastRequest(
    x=data,
    horizon=10,
    model_version="1",
    output_type="quantiles",
    quantiles=[0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
)

# Sample-based predictions
request = ToToForecastRequest(
    x=data,
    horizon=10,
    model_version="1",
    output_type="samples",
    num_samples=100
)
```

## Response Format

All forecasts return a `ForecastResponse` object:

```python
response = client.forecast(ModelName.TOTO, request)

# Access predictions based on output_type
if response.point is not None:
    predictions = response.point  # Shape: (batch_size, horizon, features)

if response.quantiles is not None:
    quantiles = response.quantiles  # Shape: (batch_size, horizon, num_quantiles)

if response.samples is not None:
    samples = response.samples  # Shape: (batch_size, horizon, num_samples)

# Access metadata
print(response.metadata)  # {'model_name': 'toto', 'model_version': '1'}
```

## Async Usage

```python
async with ForecastClient(base_url="http://localhost:8003") as client:
    response = await client.forecast_async(ModelName.FLOWSTATE, request)
    print(response.point)
```

## Examples

See the `examples/` directory for complete notebook examples:
- `flowstate_simple_example.ipynb` - Point forecasting with FlowState
- `toto_simple_example.ipynb` - Probabilistic forecasting with ToTo

## Error Handling

The SDK provides specific exception types for different error scenarios:

```python
from faim_sdk import (
    ModelNotFoundError,
    ValidationError,
    TimeoutError,
    NetworkError,
    SerializationError
)

try:
    response = client.forecast(ModelName.FLOWSTATE, request)
except ModelNotFoundError:
    print("Model or version not found")
except ValidationError:
    print("Invalid request parameters")
except TimeoutError:
    print("Request timed out")
except NetworkError:
    print("Network communication failed")
except SerializationError:
    print("Failed to serialize/deserialize data")
```

## Configuration

### Client Options

```python
client = ForecastClient(
    base_url="https://api.example.com",
    timeout=120.0,  # Request timeout in seconds
    verify_ssl=True,  # SSL certificate verification
    **httpx_kwargs  # Additional httpx.Client arguments
)
```

### Request Options

```python
request = FlowStateForecastRequest(
    x=data,
    horizon=10,
    model_version="1",
    compression="zstd",  # Options: "zstd", "lz4", None
)
```

## Requirements

- Python >= 3.10
- numpy >= 1.20.0
- pyarrow >= 10.0.0
- httpx >= 0.23.0

## License

Apache 2.0