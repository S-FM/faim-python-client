"""Visualization utilities for time-series forecasts.

This module provides production-ready plotting functions for visualizing
forecasting results. All functions use matplotlib for maximum compatibility.

To use visualization features, install with the viz extra:
    pip install faim-sdk[viz]

Typical Usage:
    >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
    >>> from faim_sdk.eval import plot_forecast
    >>> from faim_client.models import ModelName
    >>> import numpy as np
    >>>
    >>> # Generate forecast
    >>> client = ForecastClient()
    >>> request = Chronos2ForecastRequest(
    ...     x=train_data,  # shape: (32, 100, 1)
    ...     horizon=24,
    ...     output_type="point"
    ... )
    >>> response = client.forecast(ModelName.CHRONOS2, request)
    >>>
    >>> # Plot single sample from batch (omit batch dimension as documented)
    >>> fig, ax = plot_forecast(
    ...     train_data=train_data[0],  # (100, 1) - 2D array
    ...     forecast=response.point[0],  # (24, 1) - 2D array
    ...     test_data=test_data[0],  # (24, 1) - 2D array (optional)
    ...     title="Forecast for Sample 1"
    ... )
    >>> fig.savefig("forecast.png")

Available Functions:
    - plot_forecast: Plot training data, forecast, and optional test data
"""

from collections.abc import Sequence

import numpy as np

# matplotlib is an optional dependency
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None
    Axes = None


def plot_forecast(
    train_data: np.ndarray,
    forecast: np.ndarray,
    test_data: np.ndarray | None = None,
    features_on_same_plot: bool = True,
    feature_names: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot time-series forecast with training data and optional test data.

    Creates a visualization showing:
    - Historical training data (solid line)
    - Forecast predictions (dashed line)
    - Optional test/ground truth data (solid line, different color)

    **IMPORTANT**: All input arrays must be 2D (omit batch dimension).
    Input shapes: (sequence_length, features)

    Args:
        train_data: Historical training data. Shape: (train_length, features).
            Must be 2D array (batch dimension omitted).
        forecast: Forecast predictions. Shape: (horizon, features).
            Must be 2D array (batch dimension omitted).
        test_data: Optional ground truth test data. Shape: (horizon, features).
            If provided, must match forecast shape.
        features_on_same_plot: How to handle multiple features.
            - True: Plot all features on a single figure with unique colors (default)
            - False: Create subplots, one per feature
        feature_names: Optional list of feature names for legend.
            Length must match number of features.
            If None, features are named "Feature 1", "Feature 2", etc.
        title: Optional plot title. If None, uses "Time Series Forecast".
        figsize: Optional figure size (width, height) in inches.
            If None, uses matplotlib defaults or (12, 6) for subplots.
        save_path: Optional file path to save the figure.
            Supports common formats: .png, .pdf, .svg, .jpg

    Returns:
        Tuple of (Figure, Axes). For features_on_same_plot=True, Axes is a single
        Axes object. For features_on_same_plot=False, Axes is a numpy array of
        Axes objects (one per subplot).

    Raises:
        ImportError: If matplotlib is not installed. Install with: pip install faim-sdk[viz]
        TypeError: If inputs are not numpy arrays.
        ValueError: If inputs have wrong dimensions, incompatible shapes,
            are empty, or have too many features for single-plot visualization.

    Examples:
        >>> import numpy as np
        >>> from faim_sdk.eval import plot_forecast
        >>>
        >>> # Single feature example
        >>> train = np.random.randn(100, 1)  # 100 time steps, 1 feature
        >>> forecast = np.random.randn(24, 1)  # 24 forecast steps
        >>> test = np.random.randn(24, 1)  # Ground truth
        >>>
        >>> fig, ax = plot_forecast(train, forecast, test, title="Single Feature")
        >>> fig.savefig("single_feature.png")
        >>>
        >>> # Multi-feature on same plot
        >>> train = np.random.randn(100, 3)  # 3 features
        >>> forecast = np.random.randn(24, 3)
        >>> test = np.random.randn(24, 3)
        >>>
        >>> fig, ax = plot_forecast(
        ...     train,
        ...     forecast,
        ...     test,
        ...     features_on_same_plot=True,
        ...     feature_names=["Temperature", "Humidity", "Pressure"],
        ...     title="Weather Forecast"
        ... )
        >>>
        >>> # Multi-feature with subplots
        >>> fig, axes = plot_forecast(
        ...     train,
        ...     forecast,
        ...     test,
        ...     features_on_same_plot=False,
        ...     feature_names=["Temperature", "Humidity", "Pressure"],
        ...     figsize=(12, 8)
        ... )
        >>> # axes is array of 3 Axes objects
        >>> fig.savefig("weather_subplots.png")
        >>>
        >>> # Usage with FAIM SDK (remember to index batch dimension)
        >>> from faim_sdk import ForecastClient, Chronos2ForecastRequest
        >>> from faim_client.models import ModelName
        >>>
        >>> client = ForecastClient()
        >>> request = Chronos2ForecastRequest(
        ...     x=train_batch,  # (32, 100, 1)
        ...     horizon=24,
        ...     output_type="point"
        ... )
        >>> response = client.forecast(ModelName.CHRONOS2, request)
        >>>
        >>> # Plot first sample from batch
        >>> fig, ax = plot_forecast(
        ...     train_data=train_batch[0],  # (100, 1) - Remove batch dim!
        ...     forecast=response.point[0],  # (24, 1)
        ...     test_data=test_batch[0],  # (24, 1)
        ... )
        >>>
        >>> # Plot multiple samples in a loop
        >>> for i in range(5):
        ...     fig, ax = plot_forecast(
        ...         train_batch[i],
        ...         response.point[i],
        ...         test_batch[i],
        ...         title=f"Sample {i+1}",
        ...         save_path=f"forecast_sample_{i+1}.png"
        ...     )

    Notes:
        - Always pass 2D arrays (omit batch dimension when indexing SDK responses)
        - Maximum 10 features allowed when features_on_same_plot=True
        - For more than 10 features, use features_on_same_plot=False
        - Each feature gets a unique color automatically
        - Training data shown in solid lines, forecasts in dashed lines
        - Vertical line marks the train/forecast boundary
        - Legend automatically includes all features and data types

    Warnings:
        If features_on_same_plot=True and number of features > 10, raises ValueError
        to prevent cluttered visualizations.
    """
    # Check matplotlib availability
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install faim-sdk[viz] or pip install matplotlib"
        )

    # Type validation
    if not isinstance(train_data, np.ndarray):
        raise TypeError(f"train_data must be numpy.ndarray, got {type(train_data).__name__}")
    if not isinstance(forecast, np.ndarray):
        raise TypeError(f"forecast must be numpy.ndarray, got {type(forecast).__name__}")
    if test_data is not None and not isinstance(test_data, np.ndarray):
        raise TypeError(f"test_data must be numpy.ndarray or None, got {type(test_data).__name__}")

    # Shape validation
    if train_data.ndim != 2:
        raise ValueError(
            f"train_data must be 2-dimensional (sequence_length, features), got shape {train_data.shape}. "
            f"Did you forget to index the batch dimension? Use train_data[i] to select a single sample."
        )
    if forecast.ndim != 2:
        raise ValueError(
            f"forecast must be 2-dimensional (horizon, features), got shape {forecast.shape}. "
            f"Did you forget to index the batch dimension? Use forecast[i] to select a single sample."
        )
    if test_data is not None and test_data.ndim != 2:
        raise ValueError(
            f"test_data must be 2-dimensional (horizon, features), got shape {test_data.shape}. "
            f"Did you forget to index the batch dimension? Use test_data[i] to select a single sample."
        )

    # Empty validation
    if train_data.size == 0:
        raise ValueError("train_data cannot be empty")
    if forecast.size == 0:
        raise ValueError("forecast cannot be empty")

    # Extract dimensions
    train_length, train_features = train_data.shape
    horizon, forecast_features = forecast.shape

    # Feature count validation
    if train_features != forecast_features:
        raise ValueError(
            f"Feature count mismatch: train_data has {train_features} features, "
            f"forecast has {forecast_features} features"
        )

    num_features = train_features

    # Validate test_data if provided
    if test_data is not None:
        test_horizon, test_features = test_data.shape
        if test_horizon != horizon:
            raise ValueError(f"Horizon mismatch: forecast has {horizon} steps, test_data has {test_horizon} steps")
        if test_features != num_features:
            raise ValueError(
                f"Feature count mismatch: forecast has {num_features} features, test_data has {test_features} features"
            )

    # Validate feature count for single plot
    if features_on_same_plot and num_features > 10:
        raise ValueError(
            f"Cannot plot {num_features} features on same plot (maximum 10). "
            f"Use features_on_same_plot=False to create subplots instead."
        )

    # Generate feature names if not provided
    if feature_names is None:
        if num_features == 1:
            feature_names = ["Series"]
        else:
            feature_names = [f"Feature {i + 1}" for i in range(num_features)]
    else:
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match number of features ({num_features})"
            )

    # Create time indices
    train_indices = np.arange(train_length)
    forecast_indices = np.arange(train_length, train_length + horizon)

    # Set up figure
    if features_on_same_plot:
        # Single plot for all features
        if figsize is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig, ax = plt.subplots(figsize=figsize)
        axes = ax  # Return single Axes object
    else:
        # Subplots for each feature
        if figsize is None:
            figsize = (12, 3 * num_features)
        fig, axes_array = plt.subplots(num_features, 1, figsize=figsize, squeeze=False)
        axes = axes_array.flatten()  # Return array of Axes objects

    # Define color palette (matplotlib default colors)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors

    # Plot data
    if features_on_same_plot:
        # Plot all features on same axes
        for i in range(num_features):
            color = colors[i % 10]

            # Training data
            ax.plot(
                train_indices,
                train_data[:, i],
                label=f"{feature_names[i]} (train)",
                color=color,
                linewidth=2,
                alpha=0.8,
            )

            # Forecast
            ax.plot(
                forecast_indices,
                forecast[:, i],
                label=f"{feature_names[i]} (forecast)",
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

            # Test data (if provided)
            if test_data is not None:
                ax.plot(
                    forecast_indices,
                    test_data[:, i],
                    label=f"{feature_names[i]} (actual)",
                    color=color,
                    linewidth=2,
                    linestyle=":",
                    alpha=0.6,
                )

        # Add vertical line at train/forecast boundary
        ax.axvline(x=train_length, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # Labels and legend
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(title if title else "Time Series Forecast", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    else:
        # Plot each feature on separate subplot
        for i in range(num_features):
            ax_i = axes[i]
            color = colors[i % 10]

            # Training data
            ax_i.plot(
                train_indices,
                train_data[:, i],
                label="Training",
                color=color,
                linewidth=2,
                alpha=0.8,
            )

            # Forecast
            ax_i.plot(
                forecast_indices,
                forecast[:, i],
                label="Forecast",
                color=color,
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

            # Test data (if provided)
            if test_data is not None:
                ax_i.plot(
                    forecast_indices,
                    test_data[:, i],
                    label="Actual",
                    color=color,
                    linewidth=2,
                    linestyle=":",
                    alpha=0.6,
                )

            # Add vertical line at train/forecast boundary
            ax_i.axvline(x=train_length, color="gray", linestyle="-", linewidth=1, alpha=0.5)

            # Labels and legend
            ax_i.set_xlabel("Time Step", fontsize=11)
            ax_i.set_ylabel("Value", fontsize=11)
            ax_i.set_title(feature_names[i], fontsize=12, fontweight="bold")
            ax_i.legend(loc="best", fontsize=9)
            ax_i.grid(True, alpha=0.3)

        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    # Tight layout
    fig.tight_layout()

    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes
