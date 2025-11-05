"""Evaluation metrics for time-series forecasting.

This module provides production-ready implementations of standard forecasting
evaluation metrics. All functions support batch processing and flexible
aggregation options.

All metric functions expect 3D input arrays with shape:
    (batch_size, horizon, features)

Typical Usage:
    >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
    >>> from faim_sdk.eval import mse, mase, crps_from_quantiles
    >>> from faim_client.models import ModelName
    >>> import numpy as np
    >>>
    >>> # Generate forecast
    >>> client = ForecastClient()
    >>> request = Chronos2ForecastRequest(
    ...     x=train_data,  # shape: (32, 100, 1)
    ...     horizon=24,
    ...     output_type="quantiles",
    ...     quantiles=[0.1, 0.5, 0.9]
    ... )
    >>> response = client.forecast(ModelName.CHRONOS2, request)
    >>>
    >>> # Evaluate point forecasts (use median as point forecast)
    >>> point_pred = response.quantiles[:, :, 1:2]  # (32, 24, 1) - keep 3D shape
    >>> mse_score = mse(test_data, point_pred, reduction='mean')
    >>> mase_score = mase(test_data, point_pred, train_data, reduction='mean')
    >>>
    >>> # Evaluate probabilistic forecasts
    >>> crps_score = crps_from_quantiles(
    ...     test_data,
    ...     response.quantiles,
    ...     quantile_levels=[0.1, 0.5, 0.9],
    ...     reduction='mean'
    ... )

Available Metrics:
    - mse: Mean Squared Error (point forecasts)
    - mase: Mean Absolute Scaled Error (point forecasts, scale-independent)
    - crps_from_quantiles: Continuous Ranked Probability Score (probabilistic forecasts)
"""

from typing import Literal

import numpy as np

ReductionType = Literal["mean", "none"]


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reduction: ReductionType = "mean",
) -> float | np.ndarray:
    """Calculate Mean Squared Error between true and predicted values.

    MSE measures the average squared difference between predictions and ground truth.
    Lower values indicate better forecast accuracy.

    Mathematical Formula:
        MSE = (1/n) * Σ(y_true - y_pred)²

    where n is the total number of predictions (batch_size × horizon × features).

    Args:
        y_true: Ground truth values. Shape: (batch_size, horizon, features).
        y_pred: Predicted values. Shape: (batch_size, horizon, features).
        reduction: How to aggregate results across the batch dimension.
            - 'mean': Return single scalar averaged across all dimensions (default)
            - 'none': Return per-sample metrics with shape (batch_size,)

    Returns:
        MSE value(s). Returns float if reduction='mean', otherwise array of shape
        (batch_size,) containing MSE for each sample in the batch.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If inputs have different shapes, wrong number of dimensions,
            or are empty.

    Examples:
        >>> import numpy as np
        >>> from faim_sdk.eval import mse
        >>>
        >>> # Single feature, batch of 4 samples
        >>> y_true = np.array([[[1.0], [2.0], [3.0]],
        ...                     [[4.0], [5.0], [6.0]],
        ...                     [[7.0], [8.0], [9.0]],
        ...                     [[10.0], [11.0], [12.0]]])  # (4, 3, 1)
        >>> y_pred = np.array([[[1.1], [2.1], [3.1]],
        ...                     [[4.1], [5.1], [6.1]],
        ...                     [[7.1], [8.1], [9.1]],
        ...                     [[10.1], [11.1], [12.1]]])  # (4, 3, 1)
        >>>
        >>> # Overall MSE
        >>> mse(y_true, y_pred, reduction='mean')
        0.010000000000000002
        >>>
        >>> # Per-sample MSE
        >>> mse(y_true, y_pred, reduction='none')
        array([0.01, 0.01, 0.01, 0.01])
        >>>
        >>> # Multi-feature example
        >>> y_true = np.random.randn(32, 24, 5)  # 32 samples, 24 steps, 5 features
        >>> y_pred = y_true + np.random.randn(32, 24, 5) * 0.1  # Add small noise
        >>> mse_score = mse(y_true, y_pred)  # Returns scalar
        >>> print(f"MSE: {mse_score:.4f}")

    Notes:
        - MSE is sensitive to outliers due to squaring errors
        - MSE units are squared units of the original data
        - MSE is always non-negative (0 = perfect forecast)
        - For scale-independent evaluation, consider using MASE instead
    """
    # Type validation
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be numpy.ndarray, got {type(y_true).__name__}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be numpy.ndarray, got {type(y_pred).__name__}")

    # Shape validation
    if y_true.ndim != 3:
        raise ValueError(f"y_true must be 3-dimensional (batch_size, horizon, features), got shape {y_true.shape}")
    if y_pred.ndim != 3:
        raise ValueError(f"y_pred must be 3-dimensional (batch_size, horizon, features), got shape {y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    # Empty validation
    if y_true.size == 0:
        raise ValueError("y_true cannot be empty")

    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(squared_errors))
    elif reduction == "none":
        # Average over horizon and features, keep batch dimension
        return np.mean(squared_errors, axis=(1, 2))
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got '{reduction}'")


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reduction: ReductionType = "mean",
) -> float | np.ndarray:
    """Calculate Mean Absolute Error between true and predicted values.

    MAE measures the average absolute difference between predictions and ground truth.
    Lower values indicate better forecast accuracy. Unlike MSE, MAE is less sensitive
    to outliers and maintains the same units as the original data.

    Mathematical Formula:
        MAE = (1/n) * Σ|y_true - y_pred|

    where n is the total number of predictions (batch_size × horizon × features).

    Args:
        y_true: Ground truth values. Shape: (batch_size, horizon, features).
        y_pred: Predicted values. Shape: (batch_size, horizon, features).
        reduction: How to aggregate results across the batch dimension.
            - 'mean': Return single scalar averaged across all dimensions (default)
            - 'none': Return per-sample metrics with shape (batch_size,)

    Returns:
        MAE value(s). Returns float if reduction='mean', otherwise array of shape
        (batch_size,) containing MAE for each sample in the batch.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If inputs have different shapes, wrong number of dimensions,
            or are empty.

    Examples:
        >>> import numpy as np
        >>> from faim_sdk.eval import mae
        >>>
        >>> # Single feature, batch of 4 samples
        >>> y_true = np.array([[[1.0], [2.0], [3.0]],
        ...                     [[4.0], [5.0], [6.0]]])  # (2, 3, 1)
        >>> y_pred = np.array([[[1.1], [2.1], [3.1]],
        ...                     [[4.2], [5.2], [6.2]]])  # (2, 3, 1)
        >>>
        >>> # Overall MAE
        >>> mae(y_true, y_pred, reduction='mean')
        0.15
        >>>
        >>> # Per-sample MAE
        >>> mae(y_true, y_pred, reduction='none')
        array([0.1, 0.2])

    Notes:
        - MAE is less sensitive to outliers compared to MSE
        - MAE has the same units as the original data
        - MAE is always non-negative (0 = perfect forecast)
        - For scale-independent evaluation, consider using MASE instead
    """
    # Type validation
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be numpy.ndarray, got {type(y_true).__name__}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be numpy.ndarray, got {type(y_pred).__name__}")

    # Shape validation
    if y_true.ndim != 3:
        raise ValueError(f"y_true must be 3-dimensional (batch_size, horizon, features), got shape {y_true.shape}")
    if y_pred.ndim != 3:
        raise ValueError(f"y_pred must be 3-dimensional (batch_size, horizon, features), got shape {y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    # Empty validation
    if y_true.size == 0:
        raise ValueError("y_true cannot be empty")

    # Calculate absolute errors
    absolute_errors = np.abs(y_true - y_pred)

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(absolute_errors))
    elif reduction == "none":
        # Average over horizon and features, keep batch dimension
        return np.mean(absolute_errors, axis=(1, 2))
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got '{reduction}'")


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    reduction: ReductionType = "mean",
) -> float | np.ndarray:
    """Calculate Mean Absolute Scaled Error for forecast evaluation.

    MASE is a scale-independent metric that measures forecast accuracy relative
    to a naive baseline forecast. It normalizes the forecast error by the average
    error of a naive forecast on the training data.

    The naive baseline used here is the "last value" forecast: predicting each
    training value as the previous training value.

    Mathematical Formula:
        MASE = MAE(forecast) / MAE(naive baseline on training data)

        where:
        - MAE(forecast) = (1/n) * Σ|y_true - y_pred|
        - MAE(naive baseline) = (1/(T-1)) * Σ|y_train[t] - y_train[t-1]|

    Interpretation:
        - MASE < 1: Forecast is better than naive baseline
        - MASE = 1: Forecast is equivalent to naive baseline
        - MASE > 1: Forecast is worse than naive baseline

    Args:
        y_true: Ground truth values. Shape: (batch_size, horizon, features).
        y_pred: Predicted values. Shape: (batch_size, horizon, features).
        y_train: Historical training data used for baseline calculation.
            Shape: (batch_size, train_length, features).
            Should contain at least 2 time steps per sample.
        reduction: How to aggregate results across the batch dimension.
            - 'mean': Return single scalar averaged across batch (default)
            - 'none': Return per-sample metrics with shape (batch_size,)

    Returns:
        MASE value(s). Returns float if reduction='mean', otherwise array of shape
        (batch_size,) containing MASE for each sample in the batch.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If inputs have incompatible shapes, wrong dimensions,
            or training data is too short (< 2 time steps).
        RuntimeWarning: If naive baseline MAE is zero (constant training series),
            MASE will be infinite.

    Examples:
        >>> import numpy as np
        >>> from faim_sdk.eval import mase
        >>>
        >>> # Example with trend
        >>> y_train = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # (1, 5, 1)
        >>> y_true = np.array([[[6.0], [7.0], [8.0]]])  # (1, 3, 1)
        >>> y_pred = np.array([[[6.1], [7.1], [8.1]]])  # (1, 3, 1)
        >>>
        >>> # Naive baseline MAE = mean(|1-2|, |2-3|, |3-4|, |4-5|) = 1.0
        >>> # Forecast MAE = mean(|6-6.1|, |7-7.1|, |8-8.1|) = 0.1
        >>> # MASE = 0.1 / 1.0 = 0.1 (excellent, much better than naive)
        >>> mase(y_true, y_pred, y_train)
        0.1
        >>>
        >>> # Batch example with multiple samples
        >>> y_train = np.random.randn(32, 100, 5)  # 32 samples, 100 train steps
        >>> y_true = np.random.randn(32, 24, 5)   # 24 forecast steps
        >>> y_pred = np.random.randn(32, 24, 5)
        >>>
        >>> # Overall MASE
        >>> mase_score = mase(y_true, y_pred, y_train, reduction='mean')
        >>> print(f"MASE: {mase_score:.4f}")
        >>>
        >>> # Per-sample MASE
        >>> mase_per_sample = mase(y_true, y_pred, y_train, reduction='none')
        >>> print(f"MASE range: [{mase_per_sample.min():.2f}, {mase_per_sample.max():.2f}]")

    Notes:
        - MASE is scale-independent, allowing comparison across different series
        - MASE is recommended by Hyndman & Koehler (2006) for forecast evaluation
        - Unlike MAPE, MASE works well with zero or near-zero values
        - If training data is constant (naive MAE = 0), MASE will be infinite
        - For seasonal data, consider using seasonal naive baseline instead

    References:
        Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
        forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
    """
    # Type validation
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be numpy.ndarray, got {type(y_true).__name__}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be numpy.ndarray, got {type(y_pred).__name__}")
    if not isinstance(y_train, np.ndarray):
        raise TypeError(f"y_train must be numpy.ndarray, got {type(y_train).__name__}")

    # Shape validation
    if y_true.ndim != 3:
        raise ValueError(f"y_true must be 3-dimensional (batch_size, horizon, features), got shape {y_true.shape}")
    if y_pred.ndim != 3:
        raise ValueError(f"y_pred must be 3-dimensional (batch_size, horizon, features), got shape {y_pred.shape}")
    if y_train.ndim != 3:
        raise ValueError(
            f"y_train must be 3-dimensional (batch_size, train_length, features), got shape {y_train.shape}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    batch_size_true, _, features_true = y_true.shape
    batch_size_train, train_length, features_train = y_train.shape

    if batch_size_true != batch_size_train:
        raise ValueError(
            f"Batch size mismatch: y_true has {batch_size_true} samples, y_train has {batch_size_train} samples"
        )

    if features_true != features_train:
        raise ValueError(
            f"Feature count mismatch: y_true has {features_true} features, y_train has {features_train} features"
        )

    # Training data length validation
    if train_length < 2:
        raise ValueError(f"y_train must have at least 2 time steps for naive baseline, got {train_length}")

    # Empty validation
    if y_true.size == 0:
        raise ValueError("y_true cannot be empty")
    if y_train.size == 0:
        raise ValueError("y_train cannot be empty")

    # Calculate forecast MAE
    forecast_mae = np.abs(y_true - y_pred)

    # Calculate naive baseline MAE (last value forecast on training data)
    # Naive forecast: y_train[t] predicts y_train[t-1]
    naive_errors = np.abs(y_train[:, 1:, :] - y_train[:, :-1, :])
    naive_mae = np.mean(naive_errors, axis=1, keepdims=True)  # Shape: (batch_size, 1, features)

    # Warn if naive MAE is zero (constant series)
    if np.any(naive_mae == 0):
        import warnings

        warnings.warn(
            "Naive baseline MAE is zero for some samples (constant training series). "
            "MASE will be infinite for these samples.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Calculate MASE
    # Avoid division by zero by using a small epsilon where naive_mae is 0
    epsilon = 1e-10
    mase_values = forecast_mae / np.maximum(naive_mae, epsilon)

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(mase_values))
    elif reduction == "none":
        # Average over horizon and features, keep batch dimension
        return np.mean(mase_values, axis=(1, 2))
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got '{reduction}'")


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: list[float],
    reduction: ReductionType = "mean",
) -> float | np.ndarray:
    """Calculate Continuous Ranked Probability Score from quantile predictions.

    CRPS is a proper scoring rule for evaluating probabilistic forecasts.
    It generalizes the Mean Absolute Error (MAE) to probabilistic predictions
    by measuring the integrated squared difference between the predicted and
    true cumulative distribution functions.

    This implementation uses the quantile approximation method, which estimates
    CRPS from a discrete set of quantile predictions.

    Mathematical Formula (Quantile Approximation):
        CRPS ≈ Σ_i w_i * |y_true - q_i| * (2 * I(y_true < q_i) - 1 - α_i)

        where:
        - q_i is the i-th quantile prediction at level α_i
        - w_i is the weight for quantile i
        - I(·) is the indicator function

    Simplified approximation used here:
        CRPS ≈ Σ_i w_i * |y_true - q_i|

        where w_i = α_{i+1} - α_{i-1} for interior quantiles.

    Interpretation:
        - CRPS = 0: Perfect probabilistic forecast
        - Lower CRPS: Better forecast
        - CRPS reduces to MAE when only median (0.5) quantile is provided
        - CRPS penalizes both bias and miscalibration

    Args:
        y_true: Ground truth values. Shape: (batch_size, horizon, features).
        quantile_preds: Quantile predictions. Shape: (batch_size, horizon, num_quantiles).
            Quantiles should be ordered from lowest to highest level.
        quantile_levels: List of quantile levels corresponding to quantile_preds.
            Must have length equal to num_quantiles dimension.
            Values should be in [0, 1] and sorted in ascending order.
            Example: [0.1, 0.5, 0.9] for 10th, 50th, and 90th percentiles.
        reduction: How to aggregate results across the batch dimension.
            - 'mean': Return single scalar averaged across batch (default)
            - 'none': Return per-sample metrics with shape (batch_size,)

    Returns:
        CRPS value(s). Returns float if reduction='mean', otherwise array of shape
        (batch_size,) containing i for each sample in the batch.

    Raises:
        TypeError: If inputs are not numpy arrays or quantile_levels is not a list.
        ValueError: If inputs have incompatible shapes, quantile_levels are invalid,
            or arrays are empty.

    Examples:
        >>> import numpy as np
        >>> from faim_sdk.eval import crps_from_quantiles
        >>>
        >>> # Example with known quantiles
        >>> y_true = np.array([[[5.0], [6.0], [7.0]]])  # (1, 3, 1)
        >>> # Quantile predictions: 10th, 50th, 90th percentiles
        >>> quantile_preds = np.array([
        ...     [[4.5, 5.0, 5.5],  # t=0
        ...      [5.5, 6.0, 6.5],  # t=1
        ...      [6.5, 7.0, 7.5]]  # t=2
        ... ])  # (1, 3, 3)
        >>> quantile_levels = [0.1, 0.5, 0.9]
        >>>
        >>> crps_score = crps_from_quantiles(
        ...     y_true,
        ...     quantile_preds,
        ...     quantile_levels,
        ...     reduction='mean'
        ... )
        >>> print(f"CRPS: {crps_score:.4f}")
        >>>
        >>> # Typical usage with FAIM SDK response
        >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
        >>> from faim_client.models import ModelName
        >>>
        >>> client = ForecastClient(base_url="https://api.example.com")
        >>> request = Chronos2ForecastRequest(
        ...     x=train_data,  # (32, 100, 1)
        ...     horizon=24,
        ...     output_type="quantiles",
        ...     quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ... )
        >>> response = client.forecast(ModelName.CHRONOS2, request)
        >>>
        >>> # Note: response.quantiles is (batch, horizon, num_quantiles)
        >>> # but y_true is (batch, horizon, features)
        >>> # For single-feature case, this works directly:
        >>> crps = crps_from_quantiles(
        ...     test_data,  # (32, 24, 1)
        ...     response.quantiles,  # (32, 24, 5)
        ...     quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
        ... )
        >>>
        >>> # Per-sample CRPS for analysis
        >>> crps_per_sample = crps_from_quantiles(
        ...     test_data,
        ...     response.quantiles,
        ...     quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        ...     reduction='none'
        ... )
        >>> print(f"Best sample CRPS: {crps_per_sample.min():.4f}")
        >>> print(f"Worst sample CRPS: {crps_per_sample.max():.4f}")

    Notes:
        - CRPS is a proper scoring rule (incentivizes honest probabilistic forecasts)
        - More quantiles generally provide better CRPS approximation
        - CRPS is in the same units as the original data
        - For purely point forecasts, use MSE or MASE instead
        - This approximation becomes more accurate with more quantiles

    References:
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
        prediction, and estimation. Journal of the American Statistical
        Association, 102(477), 359-378.
    """
    # Type validation
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be numpy.ndarray, got {type(y_true).__name__}")
    if not isinstance(quantile_preds, np.ndarray):
        raise TypeError(f"quantile_preds must be numpy.ndarray, got {type(quantile_preds).__name__}")
    if not isinstance(quantile_levels, list):
        raise TypeError(f"quantile_levels must be a list, got {type(quantile_levels).__name__}")

    # Shape validation
    if y_true.ndim != 3:
        raise ValueError(f"y_true must be 3-dimensional (batch_size, horizon, features), got shape {y_true.shape}")
    if quantile_preds.ndim != 3:
        raise ValueError(
            f"quantile_preds must be 3-dimensional (batch_size, horizon, num_quantiles), "
            f"got shape {quantile_preds.shape}"
        )

    batch_size_true, horizon_true, features = y_true.shape
    batch_size_pred, horizon_pred, num_quantiles = quantile_preds.shape

    if batch_size_true != batch_size_pred:
        raise ValueError(
            f"Batch size mismatch: y_true has {batch_size_true} samples, quantile_preds has {batch_size_pred} samples"
        )
    if horizon_true != horizon_pred:
        raise ValueError(f"Horizon mismatch: y_true has {horizon_true} steps, quantile_preds has {horizon_pred} steps")

    # Quantile levels validation
    if len(quantile_levels) != num_quantiles:
        raise ValueError(
            f"quantile_levels length ({len(quantile_levels)}) must match num_quantiles dimension ({num_quantiles})"
        )

    if not all(0.0 <= q <= 1.0 for q in quantile_levels):
        raise ValueError(f"quantile_levels must be in [0.0, 1.0], got {quantile_levels}")

    if quantile_levels != sorted(quantile_levels):
        raise ValueError(f"quantile_levels must be sorted in ascending order, got {quantile_levels}")

    # Empty validation
    if y_true.size == 0:
        raise ValueError("y_true cannot be empty")
    if quantile_preds.size == 0:
        raise ValueError("quantile_preds cannot be empty")

    # For multi-feature forecasts, we need to ensure compatibility
    # Typically, quantile forecasts are per-feature, so we expect:
    # - If features == 1 and num_quantiles > 1: standard quantile forecast
    # - If features == num_quantiles: each quantile corresponds to a feature (unusual)
    # We'll handle the standard case where features == 1 or we average across features

    if features != 1 and features != num_quantiles:
        raise ValueError(
            f"For CRPS calculation, y_true features ({features}) should be 1 "
            f"or match num_quantiles ({num_quantiles}). "
            f"For multi-feature forecasts with quantiles, compute CRPS per feature separately."
        )

    # Calculate weights for quantile approximation
    # Weights are the differences between adjacent quantile levels
    quantile_array = np.array(quantile_levels)
    weights = np.zeros(num_quantiles)

    # Interior quantiles get weight (α_{i+1} - α_{i-1}) / 2
    if num_quantiles == 1:
        weights[0] = 1.0
    elif num_quantiles == 2:
        weights[0] = quantile_array[1] - quantile_array[0]
        weights[1] = quantile_array[1] - quantile_array[0]
    else:
        # First quantile
        weights[0] = quantile_array[1] - quantile_array[0]
        # Interior quantiles
        for i in range(1, num_quantiles - 1):
            weights[i] = (quantile_array[i + 1] - quantile_array[i - 1]) / 2.0
        # Last quantile
        weights[-1] = quantile_array[-1] - quantile_array[-2]

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Reshape for broadcasting: (1, 1, num_quantiles)
    weights_broadcast = weights.reshape(1, 1, -1)

    # For standard case where features=1, expand y_true to match quantile dimension
    if features == 1:
        # Expand y_true from (batch, horizon, 1) to (batch, horizon, num_quantiles)
        y_true_expanded = np.repeat(y_true, num_quantiles, axis=2)
    else:
        # features == num_quantiles case
        y_true_expanded = y_true

    # Calculate weighted absolute errors
    absolute_errors = np.abs(y_true_expanded - quantile_preds)
    weighted_errors = absolute_errors * weights_broadcast

    # Sum over quantiles to get CRPS per (batch, horizon) point
    crps_values = np.sum(weighted_errors, axis=2)  # Shape: (batch_size, horizon)

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(crps_values))
    elif reduction == "none":
        # Average over horizon, keep batch dimension
        return np.mean(crps_values, axis=1)
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got '{reduction}'")
