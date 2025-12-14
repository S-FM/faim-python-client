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
- **📈 Evaluation Tools**: Built-in metrics (MSE, MASE, CRPS) and visualization utilities
- **🔎 Retrieval-Augmented Inference**: Optional RAI for improved accuracy on small datasets

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

**All models require 3D input arrays:**

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

### Output Data Format

**Point Forecasts** (3D):
```python
response.point  # Shape: (batch_size, horizon, features)
```

**Quantile Forecasts** (4D):
```python
response.quantiles  # Shape: (batch_size, horizon, num_quantiles, features)
# Example: (32, 24, 5, 1) = 32 series, 24 steps ahead, 5 quantiles, 1 feature
```

### Univariate vs Multivariate

- **Chronos2**: ✅ Supports multivariate forecasting (multiple features)
- **FlowState**: ⚠️ Univariate only - automatically transforms multivariate input
- **TiRex**: ⚠️ Univariate only - automatically transforms multivariate input

When you provide multivariate input (features > 1) to FlowState or TiRex, the SDK automatically:
1. Issues a warning
2. Forecasts each feature independently
3. Reshapes the output back to your original structure

```python
# Multivariate input to FlowState
data = np.random.randn(2, 100, 3)  # 2 series, 3 features
request = FlowStateForecastRequest(x=data, horizon=24, prediction_type="mean")

# Warning: "FlowState model only supports univariate forecasting..."
response = client.forecast(request)

# Output is automatically reshaped
print(response.point.shape)  # (2, 24, 3) - original structure preserved
```

## Available Models

### FlowState

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

### Chronos 2.0

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

### TiRex

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

## Tabular Inference with LimiX

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

## Response Format

All forecasts return a `ForecastResponse` object with predictions and metadata:

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

## Evaluation & Metrics

The SDK includes a comprehensive evaluation toolkit (`faim_sdk.eval`) for measuring forecast quality with standard metrics and visualizations.

### Installation

For visualization support, install with the viz extra:

```bash
pip install faim-sdk[viz]
```

### Available Metrics

#### Mean Squared Error (MSE)

Measures average squared difference between predictions and ground truth.

```python
from faim_sdk.eval import mse

# Evaluate point forecast
mse_score = mse(test_data, response.point, reduction='mean')
print(f"MSE: {mse_score:.4f}")

# Per-sample MSE
mse_per_sample = mse(test_data, response.point, reduction='none')
print(f"MSE per sample shape: {mse_per_sample.shape}")  # (batch_size,)
```

#### Mean Absolute Scaled Error (MASE)

Scale-independent metric comparing forecast to naive baseline (better than MAPE for series with zeros).

```python
from faim_sdk.eval import mase

# MASE requires training data for baseline
mase_score = mase(test_data, response.point, train_data, reduction='mean')
print(f"MASE: {mase_score:.4f}")

# Interpretation:
# MASE < 1: Better than naive baseline
# MASE = 1: Equivalent to naive baseline
# MASE > 1: Worse than naive baseline
```

#### Continuous Ranked Probability Score (CRPS)

Proper scoring rule for probabilistic forecasts - generalizes MAE to distributions.

```python
from faim_sdk.eval import crps_from_quantiles

# Evaluate probabilistic forecast with quantiles
crps_score = crps_from_quantiles(
    test_data,
    response.quantiles,
    quantile_levels=[0.1, 0.5, 0.9],
    reduction='mean'
)
print(f"CRPS: {crps_score:.4f}")
```

### Visualization

Plot forecasts with training context and ground truth:

```python
from faim_sdk.eval import plot_forecast

# Plot single sample (remember to index batch dimension!)
fig, ax = plot_forecast(
    train_data=train_data[0],  # (seq_len, features) - 2D array
    forecast=response.point[0],  # (horizon, features) - 2D array
    test_data=test_data[0],  # (horizon, features) - optional
    title="Time Series Forecast"
)

# Save to file
fig.savefig("forecast.png", dpi=300, bbox_inches="tight")
```

#### Multi-Feature Visualization

```python
# Option 1: All features on same plot
fig, ax = plot_forecast(
    train_data[0],
    response.point[0],
    test_data[0],
    features_on_same_plot=True,
    feature_names=["Temperature", "Humidity", "Pressure"]
)

# Option 2: Separate subplots per feature
fig, axes = plot_forecast(
    train_data[0],
    response.point[0],
    test_data[0],
    features_on_same_plot=False,
    feature_names=["Temperature", "Humidity", "Pressure"]
)
```

### Complete Evaluation Example

```python
import numpy as np
from faim_sdk import ForecastClient, Chronos2ForecastRequest
from faim_sdk.eval import mse, mase, crps_from_quantiles, plot_forecast

# Initialize client
client = ForecastClient()

# Prepare data splits
train_data = np.random.randn(32, 100, 1)
test_data = np.random.randn(32, 24, 1)

# Generate forecast
request = Chronos2ForecastRequest(
    x=train_data,
    horizon=24,
    output_type="quantiles",
    quantiles=[0.1, 0.5, 0.9]
)
response = client.forecast(request)

# Evaluate point forecast (use median)
point_pred = response.quantiles[:, :, 1:2]  # Extract median, keep 3D shape
mse_score = mse(test_data, point_pred)
mase_score = mase(test_data, point_pred, train_data)

# Evaluate probabilistic forecast
crps_score = crps_from_quantiles(
    test_data,
    response.quantiles,
    quantile_levels=[0.1, 0.5, 0.9]
)

print(f"MSE: {mse_score:.4f}")
print(f"MASE: {mase_score:.4f}")
print(f"CRPS: {crps_score:.4f}")

# Visualize best and worst predictions
mse_per_sample = mse(test_data, point_pred, reduction='none')
best_idx = np.argmin(mse_per_sample)
worst_idx = np.argmax(mse_per_sample)

fig1, ax1 = plot_forecast(
    train_data[best_idx],
    point_pred[best_idx],
    test_data[best_idx],
    title=f"Best Forecast (MSE: {mse_per_sample[best_idx]:.4f})"
)
fig1.savefig("best_forecast.png")

fig2, ax2 = plot_forecast(
    train_data[worst_idx],
    point_pred[worst_idx],
    test_data[worst_idx],
    title=f"Worst Forecast (MSE: {mse_per_sample[worst_idx]:.4f})"
)
fig2.savefig("worst_forecast.png")
```

## Error Handling

The SDK provides **machine-readable error codes** for robust error handling:

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

- **`toy_example.ipynb`** - A toy example showing how to get started with FAIM and generate both point and probabilistic forecasts.

## Requirements

- Python >= 3.10
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

- **Email**: support@faim.it.com

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Citation

If you use FAIM in your research, please cite:

```bibtex
@software{faim_sdk,
  title = {FAIM SDK: Foundation AI Models for Time Series Forecasting},
  author = {FAIM Team},
  year = {2024},
  url = {https://github.com/S-FM/faim-python-client}
}
```
