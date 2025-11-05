"""Evaluation tools for time-series forecasting.

This subpackage provides metrics and visualization utilities for evaluating
forecast quality. All functions work seamlessly with FAIM SDK responses.

Modules:
    - metrics: MSE, MASE, and CRPS metrics for forecast evaluation
    - visualization: plot_forecast for visualizing forecasts

Quick Start - Metrics:
    >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
    >>> from faim_sdk.eval import mse, mase, crps_from_quantiles
    >>> from faim_client.models import ModelName
    >>> import numpy as np
    >>>
    >>> # Generate forecast
    >>> client = ForecastClient()
    >>> request = Chronos2ForecastRequest(
    ...     x=train_data,  # (32, 100, 1)
    ...     horizon=24,
    ...     output_type="quantiles",
    ...     quantiles=[0.1, 0.5, 0.9]
    ... )
    >>> response = client.forecast(ModelName.CHRONOS2, request)
    >>>
    >>> # Evaluate point forecast (use median)
    >>> point_pred = response.quantiles[:, :, 1:2]  # Keep 3D shape
    >>> mse_score = mse(test_data, point_pred)
    >>> mase_score = mase(test_data, point_pred, train_data)
    >>>
    >>> # Evaluate probabilistic forecast
    >>> crps_score = crps_from_quantiles(
    ...     test_data,
    ...     response.quantiles,
    ...     quantile_levels=[0.1, 0.5, 0.9]
    ... )

Quick Start - Visualization:
    >>> from faim_sdk.eval import plot_forecast
    >>>
    >>> # Plot single sample (remember to index batch dimension!)
    >>> fig, ax = plot_forecast(
    ...     train_data=train_data[0],  # (100, 1) - 2D array
    ...     forecast=response.point[0],  # (24, 1) - 2D array
    ...     test_data=test_data[0],  # (24, 1) - 2D array
    ...     title="Forecast Visualization"
    ... )
    >>> fig.savefig("forecast.png")

Available Metrics:
    - mse: Mean Squared Error (point forecasts)
    - mae: Mean Absolute Error (point forecasts)
    - mase: Mean Absolute Scaled Error (scale-independent)
    - crps_from_quantiles: Continuous Ranked Probability Score (probabilistic)

Available Visualization:
    - plot_forecast: Plot training data, forecast, and optional test data

Installation:
    For visualization support, install with the viz extra:
        pip install faim-sdk[viz]

    Or install matplotlib separately:
        pip install matplotlib
"""

from .metrics import crps_from_quantiles, mae, mase, mse
from .visualization import plot_forecast

__all__ = [
    # Metrics
    "mse",
    "mae",
    "mase",
    "crps_from_quantiles",
    # Visualization
    "plot_forecast",
]
