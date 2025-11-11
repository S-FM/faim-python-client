"""Quick start example for FAIM SDK."""

import numpy as np

from faim_sdk import Chronos2ForecastRequest, ForecastClient

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
    quantiles=[0.1, 0.5, 0.9],  # 10th, 50th (median), 90th percentiles
)

# Generate forecast - model inferred automatically from request type
response = client.forecast(request)

# Access predictions
print(response.quantiles.shape)  # (32, 24, 3, 1)
print(response.metadata)  # Model version, inference time, etc.
