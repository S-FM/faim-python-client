# FAIM SDK

[![PyPI version](https://badge.fury.io/py/faim-sdk.svg)](https://badge.fury.io/py/faim-sdk)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Production-ready Python SDK for FAIM (Foundation AI Models) - a unified platform for time-series forecasting and tabular inference powered by foundation models.

## Features

- **🚀 Multiple Foundation Models**:
  - **Time-Series**: FlowState, Amazon Chronos 2.0, TiRex
  - **Tabular**: LimiX (classification & regression)
- **🔒 Type-Safe API**: Full type hints with Pydantic validation
- **⚡ High Performance**: Optimized Apache Arrow serialization with zero-copy operations
- **🎯 Probabilistic & Deterministic**: Point forecasts, quantiles, samples, and probabilistic predictions
- **🔄 Async Support**: Built-in async/await support for concurrent requests
- **📊 Rich Error Handling**: Machine-readable error codes with detailed diagnostics
- **🧪 Battle-Tested**: Production-ready with comprehensive error handling

## Installation

```bash
pip install faim-sdk
```

## Authentication

Get your API key at **[https://faim.it.com/](https://faim.it.com/)**

```python
from faim_sdk import ForecastClient

# Initialize client with your API key
client = ForecastClient(api_key="your-api-key")
```

## Quick Start

```python
import numpy as np
from faim_sdk import ForecastClient, Chronos2ForecastRequest

# Initialize client
client = ForecastClient(api_key="your-api-key")

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

# Generate forecast - model inferred automatically from request type
response = client.forecast(request)

# Access predictions
print(response.quantiles.shape)  # (32, 24, 3, 1)
print(response.metadata)  # Model version, inference time, etc.
```

## Input/Output Format

### Input Data Format

#### Time-Series Models (FlowState, Chronos2, TiRex)

**All time-series models require 3D input arrays:**

```python
# Shape: (batch_size, sequence_length, features)
x = np.array([
    [[1.0], [2.0], [3.0]],  # Series 1
    [[4.0], [5.0], [6.0]]   # Series 2
])  # Shape: (2, 3, 1)
```

- **batch_size**: Number of independent time series
- **sequence_length**: Historical data points (context window)
- **features**: Number of variables per time step (use 1 for univariate)

**Important**: 2D input will raise a validation error. Always provide 3D arrays.

#### Tabular Models (LimiX)

**Tabular models require 2D input arrays:**

```python
# Shape: (n_samples, n_features)
X_train = np.array([
    [1.0, 2.0, 3.0],  # Sample 1
    [4.0, 5.0, 6.0],  # Sample 2
])  # Shape: (2, 3)
```

- **n_samples**: Number of training/test samples
- **n_features**: Number of input features per sample

### Output Data Format

#### Time-Series Output

**Point Forecasts** (3D):
```python
response.point  # Shape: (batch_size, horizon, features)
```

**Quantile Forecasts** (4D):
```python
response.quantiles  # Shape: (batch_size, horizon, num_quantiles, features)
# Example: (32, 24, 5, 1) = 32 series, 24 steps ahead, 5 quantiles, 1 feature
```

#### Tabular Output

**Predictions** (1D):
```python
response.predictions  # Shape: (n_samples,)
# Classification: class labels or indices
# Regression: continuous values
```

**Classification Probabilities** (2D):
```python
response.probabilities  # Shape: (n_samples, n_classes) - classification only
# Probability for each class
```

### Univariate vs Multivariate (Time-Series Only)

- **Chronos2**: ✅ Supports multivariate forecasting (multiple features)
- **FlowState**: ⚠️ Univariate only - automatically transforms multivariate input
- **TiRex**: ⚠️ Univariate only - automatically transforms multivariate input

## Available Models

### Model Selection Guide

Choose your client and model based on your task:

| Task | Client | Models | Input | Output |
|------|--------|--------|-------|--------|
| **Time-Series Forecasting** | `ForecastClient` | FlowState, Chronos2, TiRex | 3D: `(batch, seq_len, features)` | 3D/4D point/quantiles |
| **Tabular Classification** | `TabularClient` | LimiX | 2D: `(n_samples, n_features)` | 1D predictions + 2D probabilities |
| **Tabular Regression** | `TabularClient` | LimiX | 2D: `(n_samples, n_features)` | 1D continuous predictions |

### Time-Series Models

#### FlowState

```python
from faim_sdk import FlowStateForecastRequest

request = FlowStateForecastRequest(
    x=data,
    horizon=24,
    model_version="latest",
    output_type="point",
    scale_factor=1.0,  # Optional: normalization factor, for details check: https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1
    prediction_type="mean"  # Options: "mean", "median"
)

response = client.forecast(request)
print(response.point.shape)  # (batch_size, 24, features)
```

#### Chronos 2.0

```python
from faim_sdk import Chronos2ForecastRequest

# Quantile-based probabilistic forecast
request = Chronos2ForecastRequest(
    x=data,
    horizon=24,
    output_type="quantiles",
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]  # Full distribution
)

response = client.forecast(request)
print(response.quantiles.shape)  # (batch_size, 24, 5)
```

#### TiRex

```python
from faim_sdk import TiRexForecastRequest

request = TiRexForecastRequest(
    x=data,
    horizon=24,
    output_type="point"
)

response = client.forecast(request)
print(response.point.shape)  # (batch_size, 24, features)
```

### LimiX

The SDK also supports **LimiX**, a foundation model for tabular classification and regression:

```python
from faim_sdk import TabularClient, LimiXPredictRequest
import numpy as np

# Initialize tabular client
client = TabularClient(api_key="your-api-key")

# Prepare tabular data (2D arrays)
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = np.random.randint(0, 2, 100).astype(np.float32)
X_test = np.random.randn(20, 10).astype(np.float32)

# Create classification request
request = LimiXPredictRequest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    task_type="Classification",  # or "Regression"
    use_retrieval=False  # Set to True for retrieval-augmented inference
)

# Generate predictions
response = client.predict(request)
print(response.predictions.shape)   # (20,)
print(response.probabilities.shape)  # (20, n_classes) - classification only
```

### Classification Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Convert to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)

# Create and send request
request = LimiXPredictRequest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    task_type="Classification"
)

response = client.predict(request)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, response.predictions.astype(int))
print(f"Accuracy: {accuracy:.4f}")
```

### Regression Example

```python
from sklearn.datasets import fetch_california_housing

# Load dataset
house_data = fetch_california_housing()
X, y = house_data.data, house_data.target

# Split data (50/50 for demo)
split_idx = len(X) // 2
X_train, X_test = X[:split_idx].astype(np.float32), X[split_idx:].astype(np.float32)
y_train, y_test = y[:split_idx].astype(np.float32), y[split_idx:].astype(np.float32)

# Create and send request
request = LimiXPredictRequest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    task_type="Regression"
)

response = client.predict(request)

# Evaluate
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, response.predictions))
print(f"RMSE: {rmse:.4f}")
```

### Retrieval-Augmented Inference

For better accuracy on small datasets, enable retrieval-augmented inference:

```python
request = LimiXPredictRequest(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    task_type="Classification",
    use_retrieval=True  # Enable RAI (slower but more accurate)
)

response = client.predict(request)
```

## Response Format (Time-Series Forecasting)

Time-series forecasts return a `ForecastResponse` object with predictions and metadata:

```python
response = client.forecast(request)

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

The SDK provides **error codes** for robust error handling:

```python
from faim_sdk import (
    ForecastClient,
    Chronos2ForecastRequest,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ErrorCode
)

try:
    request = Chronos2ForecastRequest(x=data, horizon=24, quantiles=[0.1, 0.5, 0.9])
    response = client.forecast(request)

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

async def forecast_multiple_series():
    client = ForecastClient(
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
            client.forecast_async(req)
            for req in requests
        ]
        responses = await asyncio.gather(*tasks)

    return responses

# Run async forecasts
responses = asyncio.run(forecast_multiple_series())
```

## Examples

See the `examples/` directory for complete Jupyter notebook examples:

### Time-Series Forecasting
- **`toy_example.ipynb`** - Get started with FAIM and generate both point and probabilistic forecasts
- **`airpassengers_dataset.ipynb`** - End-to-end example with AirPassengers dataset

### Tabular Inference with LimiX
- **`limix_classification_example.ipynb`** - Binary classification on breast cancer dataset
  
- **`limix_regression_example.ipynb`** - Regression on California housing dataset
  
## Requirements

- Python >= 3.10
- numpy >= 1.26.0
- pyarrow >= 11.0.0
- httpx >= 0.23.0
- pydantic >= 2.0.0

## Performance Tips

### Time-Series Forecasting

1. **Batch Processing**: Process multiple time series in a single request for optimal throughput
   ```python
   # Good: Single request with 32 series
   data = np.random.randn(32, 100, 1)

   # Less efficient: 32 separate requests
   # for series in data: client.forecast(...)
   ```

2. **Compression**: Use `compression="zstd"` for large payloads (default, recommended)

### General (All Models)

4. **Connection Pooling**: Reuse client instances across requests instead of creating new ones

## Support

- **Email**: info@faim.it.com

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Citation

If you use FAIM in your research, please cite:

```bibtex
@software{faim_sdk,
  title = {FAIM SDK: Foundation AI Models for Time Series Forecasting},
  author = {FAIM Team},
  year = {2025},
  url = {https://github.com/S-FM/faim-python-client}
}
```
