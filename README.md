# FAIM Client

Python SDK for the FAIM (Foundation AI Models) inference platform - time-series forecasting using foundation models on structured data.

## Installation

```bash
# Using Poetry (recommended)
poetry add faim-client

# Using pip
pip install faim-client

# Development installation
pip install -e .
```

## Quick Start

```python
from faim_sdk import ForecastClient, FlowStateForecastRequest
from faim_client.models import ModelName
import numpy as np

# Initialize client
client = ForecastClient(base_url="https://api.faim.example.com")

# Create forecast request
request = FlowStateForecastRequest(
    x=np.random.rand(1, 100, 1),  # Shape: (batch_size, seq_len, features)
    horizon=10,                    # Forecast 10 steps ahead
    model_version="latest"
)

# Get predictions
response = client.forecast(ModelName.FLOWSTATE, request)
print(response.predictions.shape)  # (1, 10, 1)
```

## Features

- **Type-Safe API**: Model-specific request classes with full IDE autocomplete
- **Automatic Serialization**: Efficient Apache Arrow serialization with zero-copy optimizations
- **Error Handling**: Comprehensive exception hierarchy for precise error handling
- **Async Support**: Both sync (`forecast()`) and async (`forecast_async()`) methods
- **Production-Ready**: Logging, observability, and proper resource management

## Models

### FlowState - Point Forecasting

```python
from faim_sdk import FlowStateForecastRequest

request = FlowStateForecastRequest(
    x=data,
    horizon=10,
    scale_factor=1.0,           # Optional: normalization factor
    prediction_type="point"      # Optional: prediction mode
)

response = client.forecast(ModelName.FLOWSTATE, request)
```

### ToTo - Probabilistic Forecasting

```python
from faim_sdk import ToToForecastRequest

request = ToToForecastRequest(
    x=data,
    horizon=10,
    padding_mask=mask,           # Optional: for variable-length sequences
    id_mask=ids,                 # Optional: for multi-series forecasting
    quantiles=[0.1, 0.5, 0.9]    # Optional: quantile predictions
)

response = client.forecast(ModelName.TOTO, request)
print(response.quantiles.shape)  # (batch_size, horizon, num_quantiles)
```

## Examples

See the `examples/` directory for complete notebooks:
- `flowstate_simple_example.ipynb`: Forecasting with AirPassengers dataset

## Architecture

This package contains two layers:

### `faim_sdk/` - Production SDK (Recommended)
High-level, type-safe API with automatic Arrow serialization and error handling.
Use this for application development.

### `faim_client/` - Generated Client (Low-level)
Auto-generated from OpenAPI spec. Use only for advanced customization.

## Development

### Regenerating the Client

When the OpenAPI spec changes:

```bash
openapi-python-client generate \
  --path openapi.json \
  --config client.config.yaml \
  --overwrite \
  --meta none
```

**Important**: Only regenerate `faim_client/`. The `faim_sdk/` code is manually maintained.

### Building and Publishing

```bash
# Build package
poetry build

# Publish to PyPI
poetry publish --build

# Publish to private repository
poetry config repositories.<repo-name> <url>
poetry publish --build -r <repo-name>
```

## Advanced Usage

### Async Forecasting

```python
async with ForecastClient(base_url="...") as client:
    response = await client.forecast_async(ModelName.FLOWSTATE, request)
```

### Custom Configuration

```python
client = ForecastClient(
    base_url="https://api.example.com",
    timeout=300.0,                          # Custom timeout
    verify_ssl=True,                        # SSL verification
    httpx_args={                            # Additional httpx config
        "limits": httpx.Limits(max_connections=10)
    }
)
```

### Error Handling

```python
from faim_sdk.exceptions import (
    ModelNotFoundError,
    ValidationError,
    PayloadTooLargeError,
    TimeoutError
)

try:
    response = client.forecast(ModelName.FLOWSTATE, request)
except ModelNotFoundError:
    print("Model or version doesn't exist")
except ValidationError as e:
    print(f"Invalid parameters: {e.response}")
except TimeoutError:
    print("Request timed out - consider reducing batch size")
```

## Performance

The SDK uses optimized Apache Arrow serialization:
- Zero-copy operations when possible
- Configurable compression (zstd, lz4, none)
- Efficient batch processing
- Minimal memory overhead

## Documentation

- See `CLAUDE.md` for architecture details and development guidance
- API documentation: [Link to your docs]
- Backend repository: [Link to backend]

## License

[Your License]