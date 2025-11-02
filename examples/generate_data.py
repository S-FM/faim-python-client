"""
Synthetic time series data generation for FAIM SDK examples.

This module provides functions to generate synthetic time series data with clear patterns
for testing and demonstrating forecasting capabilities.
"""

import numpy as np
from typing import Tuple


def generate_linear_trend_series(
    batch_size: int = 1,
    context_length: int = 256,
    trend_slope: float = 0.05,
    noise_std: float = 0.5,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate simple time series with linear trend and Gaussian noise.

    Pattern: y(t) = trend_slope * t + noise
    This creates an easily forecastable upward or downward trend.

    Args:
        batch_size: Number of independent time series to generate
        context_length: Length of each time series
        trend_slope: Slope of the linear trend (positive for upward, negative for downward)
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (batch_size, context_length, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate time indices
    t = np.arange(context_length)

    # Create linear trend
    trend = trend_slope * t

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, size=(batch_size, context_length))

    # Combine trend and noise
    series = trend[np.newaxis, :] + noise

    # Reshape to (batch_size, context_length, 1)
    return series[:, :, np.newaxis]


def generate_correlated_multi_series(
    batch_size: int = 1,
    context_length: int = 256,
    correlation: float = 0.8,
    seasonal_period: int = 24,
    seasonal_amplitude: float = 2.0,
    trend_slope: float = 0.02,
    noise_std: float = 0.3,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate two correlated time series with seasonal patterns.

    Pattern:
    - Series 1: trend + seasonality + noise
    - Series 2: correlation * Series1 + (1-correlation) * independent_pattern + noise

    This creates two related time series where forecasting one can help predict the other.

    Args:
        batch_size: Number of independent time series pairs to generate
        context_length: Length of each time series
        correlation: Correlation coefficient between the two series (0 to 1)
        seasonal_period: Period of seasonal component (e.g., 24 for daily pattern)
        seasonal_amplitude: Amplitude of seasonal oscillation
        trend_slope: Slope of the linear trend
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (batch_size, context_length, 2) - two correlated features
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate time indices
    t = np.arange(context_length)

    # Create shared components
    trend = trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)

    # Generate first series
    series1_base = trend + seasonal
    noise1 = np.random.normal(0, noise_std, size=(batch_size, context_length))
    series1 = series1_base[np.newaxis, :] + noise1

    # Generate second series correlated with first
    # Create independent pattern for mixing
    independent_seasonal = seasonal_amplitude * np.cos(2 * np.pi * t / (seasonal_period * 1.5))
    independent_base = trend * 0.8 + independent_seasonal
    noise2 = np.random.normal(0, noise_std, size=(batch_size, context_length))

    # Mix correlated and independent components
    series2 = (
        correlation * series1 +
        (1 - correlation) * (independent_base[np.newaxis, :] + noise2)
    )

    # Stack into (batch_size, context_length, 2)
    multi_series = np.stack([series1, series2], axis=-1)

    return multi_series


def generate_heavy_payload(
    batch_size: int = 100,
    context_length: int = 2048,
    num_features: int = 1,
    pattern_type: str = "mixed",
    noise_std: float = 0.4,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate large batch of time series for stress testing and performance evaluation.

    Pattern options:
    - "mixed": Combination of trends, seasonality, and cyclic patterns
    - "seasonal": Pure seasonal patterns with varying periods
    - "trend": Linear trends with varying slopes

    Args:
        batch_size: Number of independent time series (default 100 for heavy load)
        context_length: Length of each time series (default 2048 for long sequences)
        num_features: Number of features per time step
        pattern_type: Type of pattern to generate ("mixed", "seasonal", or "trend")
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (batch_size, context_length, num_features)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate time indices
    t = np.arange(context_length)

    # Initialize output array
    series = np.zeros((batch_size, context_length, num_features))

    for i in range(batch_size):
        for f in range(num_features):
            if pattern_type == "mixed":
                # Combine trend, multiple seasonal components, and noise
                trend = (0.01 + 0.02 * np.random.rand()) * t

                # Primary seasonal component
                period1 = 100 + np.random.randint(-20, 20)
                amp1 = 1.0 + 2.0 * np.random.rand()
                seasonal1 = amp1 * np.sin(2 * np.pi * t / period1 + 2 * np.pi * np.random.rand())

                # Secondary seasonal component
                period2 = 500 + np.random.randint(-100, 100)
                amp2 = 0.5 + 1.0 * np.random.rand()
                seasonal2 = amp2 * np.cos(2 * np.pi * t / period2 + 2 * np.pi * np.random.rand())

                # Combine components
                base = trend + seasonal1 + seasonal2

            elif pattern_type == "seasonal":
                # Pure seasonal patterns with varying periods
                period = 50 + np.random.randint(0, 200)
                amplitude = 1.0 + 3.0 * np.random.rand()
                phase = 2 * np.pi * np.random.rand()
                base = amplitude * np.sin(2 * np.pi * t / period + phase)

            elif pattern_type == "trend":
                # Linear trends with varying slopes and intercepts
                slope = -0.05 + 0.1 * np.random.rand()
                intercept = -5.0 + 10.0 * np.random.rand()
                base = slope * t + intercept

            else:
                raise ValueError(f"Unknown pattern_type: {pattern_type}")

            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, size=context_length)

            # Assign to output array
            series[i, :, f] = base + noise

    return series


def generate_test_suite(seed: int = 42) -> dict[str, np.ndarray]:
    """
    Generate a complete test suite with all data patterns.

    Args:
        seed: Random seed for reproducibility

    Returns:
        dict: Dictionary containing all generated datasets with descriptive keys
    """
    return {
        # Simple patterns
        "linear_trend_single": generate_linear_trend_series(
            batch_size=1, context_length=256, seed=seed
        ),
        "linear_trend_batch": generate_linear_trend_series(
            batch_size=16, context_length=256, seed=seed
        ),

        # Correlated multi-series
        "correlated_multi_single": generate_correlated_multi_series(
            batch_size=1, context_length=256, seed=seed
        ),
        "correlated_multi_batch": generate_correlated_multi_series(
            batch_size=16, context_length=256, seed=seed
        ),

        # Heavy payloads
        "heavy_mixed": generate_heavy_payload(
            batch_size=100, context_length=2048, pattern_type="mixed", seed=seed
        ),
        "heavy_seasonal": generate_heavy_payload(
            batch_size=100, context_length=2048, pattern_type="seasonal", seed=seed
        ),
        "heavy_trend": generate_heavy_payload(
            batch_size=100, context_length=2048, pattern_type="trend", seed=seed
        ),
    }


if __name__ == "__main__":
    # Example usage and validation
    print("Generating synthetic time series data...\n")

    # Test linear trend series
    linear_data = generate_linear_trend_series(batch_size=5, context_length=100, seed=42)
    print(f"Linear trend series shape: {linear_data.shape}")
    print(f"  Expected: (5, 100, 1)")
    print(f"  Min: {linear_data.min():.2f}, Max: {linear_data.max():.2f}, Mean: {linear_data.mean():.2f}\n")

    # Test correlated multi-series
    multi_data = generate_correlated_multi_series(batch_size=3, context_length=100, seed=42)
    print(f"Correlated multi-series shape: {multi_data.shape}")
    print(f"  Expected: (3, 100, 2)")
    # Calculate correlation between the two series
    corr = np.corrcoef(multi_data[0, :, 0], multi_data[0, :, 1])[0, 1]
    print(f"  Correlation between series: {corr:.2f}\n")

    # Test heavy payload
    heavy_data = generate_heavy_payload(batch_size=100, context_length=2048, seed=42)
    print(f"Heavy payload shape: {heavy_data.shape}")
    print(f"  Expected: (100, 2048, 1)")
    print(f"  Memory size: {heavy_data.nbytes / 1024 / 1024:.2f} MB\n")

    # Generate full test suite
    print("Generating full test suite...")
    test_suite = generate_test_suite(seed=42)
    print(f"Generated {len(test_suite)} datasets:")
    for name, data in test_suite.items():
        print(f"  {name}: {data.shape}")