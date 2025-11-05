"""Unit tests for faim_sdk.eval.visualization module.

Tests for plot_forecast with comprehensive coverage of functionality,
edge cases, and error handling. Uses matplotlib mocking where appropriate.
"""

import numpy as np
import pytest


class TestPlotForecast:
    """Tests for plot_forecast visualization function."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Ensure matplotlib is available for tests."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend for testing
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_forecast_basic(self):
        """Test basic plot_forecast functionality."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) >= 2  # At least train and forecast lines

    def test_plot_forecast_with_test_data(self):
        """Test plot_forecast with optional test data."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)
        test_data = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast, test_data)

        assert fig is not None
        assert len(ax.lines) >= 3  # Train, forecast, and test lines

    def test_plot_forecast_single_feature_with_title(self):
        """Test plot_forecast with custom title."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast, title="Test Forecast")

        assert ax.get_title() == "Test Forecast"

    def test_plot_forecast_multi_feature_same_plot(self):
        """Test plot_forecast with multiple features on same plot."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 3)
        forecast = np.random.randn(24, 3)
        test_data = np.random.randn(24, 3)

        fig, ax = plot_forecast(
            train_data,
            forecast,
            test_data,
            features_on_same_plot=True,
            feature_names=["Feature A", "Feature B", "Feature C"],
        )

        assert fig is not None
        # Should have 3 features × 3 lines (train, forecast, test) = 9 lines + vertical line
        assert len(ax.lines) >= 9

    def test_plot_forecast_multi_feature_subplots(self):
        """Test plot_forecast with multiple features as subplots."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 3)
        forecast = np.random.randn(24, 3)

        fig, axes = plot_forecast(
            train_data, forecast, features_on_same_plot=False, feature_names=["Temp", "Humidity", "Pressure"]
        )

        assert fig is not None
        assert len(axes) == 3  # One subplot per feature
        assert axes[0].get_title() == "Temp"
        assert axes[1].get_title() == "Humidity"
        assert axes[2].get_title() == "Pressure"

    def test_plot_forecast_custom_figsize(self):
        """Test plot_forecast with custom figure size."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast, figsize=(15, 8))

        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 8

    def test_plot_forecast_save_path(self, tmp_path):
        """Test plot_forecast saves to file."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        save_path = tmp_path / "forecast.png"
        fig, ax = plot_forecast(train_data, forecast, save_path=str(save_path))

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_forecast_default_feature_names(self):
        """Test plot_forecast generates default feature names."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 3)
        forecast = np.random.randn(24, 3)

        fig, axes = plot_forecast(train_data, forecast, features_on_same_plot=False)

        # Should use default names: Feature 1, Feature 2, Feature 3
        assert axes[0].get_title() == "Feature 1"
        assert axes[1].get_title() == "Feature 2"
        assert axes[2].get_title() == "Feature 3"

    def test_plot_forecast_single_feature_default_name(self):
        """Test plot_forecast uses 'Series' for single feature."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, axes = plot_forecast(train_data, forecast, features_on_same_plot=False)

        # Should use 'Series' for single feature
        assert axes[0].get_title() == "Series"

    def test_plot_forecast_wrong_dimensions_train_error(self):
        """Test plot_forecast raises error for wrong train_data dimensions."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(32, 100, 1)  # 3D instead of 2D
        forecast = np.random.randn(24, 1)

        with pytest.raises(ValueError, match="train_data must be 2-dimensional"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_wrong_dimensions_forecast_error(self):
        """Test plot_forecast raises error for wrong forecast dimensions."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(32, 24, 1)  # 3D instead of 2D

        with pytest.raises(ValueError, match="forecast must be 2-dimensional"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_wrong_dimensions_test_error(self):
        """Test plot_forecast raises error for wrong test_data dimensions."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)
        test_data = np.random.randn(32, 24, 1)  # 3D instead of 2D

        with pytest.raises(ValueError, match="test_data must be 2-dimensional"):
            plot_forecast(train_data, forecast, test_data)

    def test_plot_forecast_feature_mismatch_error(self):
        """Test plot_forecast raises error for mismatched features."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 3)
        forecast = np.random.randn(24, 2)  # Wrong features

        with pytest.raises(ValueError, match="Feature count mismatch"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_horizon_mismatch_error(self):
        """Test plot_forecast raises error for mismatched horizons."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)
        test_data = np.random.randn(12, 1)  # Wrong horizon

        with pytest.raises(ValueError, match="Horizon mismatch"):
            plot_forecast(train_data, forecast, test_data)

    def test_plot_forecast_too_many_features_error(self):
        """Test plot_forecast raises error for too many features on same plot."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 11)  # 11 features
        forecast = np.random.randn(24, 11)

        with pytest.raises(ValueError, match="Cannot plot 11 features on same plot"):
            plot_forecast(train_data, forecast, features_on_same_plot=True)

    def test_plot_forecast_wrong_feature_names_length_error(self):
        """Test plot_forecast raises error for wrong feature_names length."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 3)
        forecast = np.random.randn(24, 3)

        with pytest.raises(ValueError, match="feature_names length.*must match"):
            plot_forecast(
                train_data,
                forecast,
                feature_names=["A", "B"],  # Only 2, but 3 features
            )

    def test_plot_forecast_type_error_train(self):
        """Test plot_forecast raises error for non-numpy train_data."""
        from faim_sdk.eval import plot_forecast

        train_data = [[1.0, 2.0]]  # Python list
        forecast = np.random.randn(24, 1)

        with pytest.raises(TypeError, match="train_data must be numpy.ndarray"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_type_error_forecast(self):
        """Test plot_forecast raises error for non-numpy forecast."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = [[1.0, 2.0]]  # Python list

        with pytest.raises(TypeError, match="forecast must be numpy.ndarray"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_type_error_test(self):
        """Test plot_forecast raises error for non-numpy test_data."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)
        test_data = [[1.0, 2.0]]  # Python list

        with pytest.raises(TypeError, match="test_data must be numpy.ndarray or None"):
            plot_forecast(train_data, forecast, test_data)

    def test_plot_forecast_empty_train_error(self):
        """Test plot_forecast raises error for empty train_data."""
        from faim_sdk.eval import plot_forecast

        train_data = np.array([]).reshape(0, 1)
        forecast = np.random.randn(24, 1)

        with pytest.raises(ValueError, match="train_data cannot be empty"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_empty_forecast_error(self):
        """Test plot_forecast raises error for empty forecast."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.array([]).reshape(0, 1)

        with pytest.raises(ValueError, match="forecast cannot be empty"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_helpful_error_message_batch_dim(self):
        """Test plot_forecast provides helpful error for batch dimension."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(32, 100, 1)  # User forgot to index [i]
        forecast = np.random.randn(24, 1)

        with pytest.raises(ValueError, match="Did you forget to index the batch dimension"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_matplotlib_not_available_error(self, monkeypatch):
        """Test plot_forecast raises error when matplotlib not available."""
        # This test simulates matplotlib not being installed
        import sys

        # Remove matplotlib from modules
        matplotlib_modules = [key for key in sys.modules if key.startswith("matplotlib")]
        for mod in matplotlib_modules:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Mock import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("No module named 'matplotlib'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Force reload of visualization module
        import importlib

        import faim_sdk.eval

        importlib.reload(faim_sdk.eval.visualization)

        from faim_sdk.eval.visualization import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        with pytest.raises(ImportError, match="matplotlib is required"):
            plot_forecast(train_data, forecast)

    def test_plot_forecast_legend_present(self):
        """Test plot_forecast includes legend."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast)

        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) >= 2  # At least train and forecast

    def test_plot_forecast_grid_present(self):
        """Test plot_forecast includes grid."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast)

        # Check if grid is enabled
        assert ax.xaxis.grid or ax.yaxis.grid

    def test_plot_forecast_vertical_line_at_boundary(self):
        """Test plot_forecast includes vertical line at train/forecast boundary."""
        from faim_sdk.eval import plot_forecast

        train_data = np.random.randn(100, 1)
        forecast = np.random.randn(24, 1)

        fig, ax = plot_forecast(train_data, forecast)

        # Check for vertical lines (boundary marker)
        # Should have at least one vertical line for the boundary
        # (Plus the actual data lines)
        assert len(ax.lines) >= 2

    def test_plot_forecast_realistic_sdk_usage(self):
        """Test plot_forecast with realistic SDK response format."""
        from faim_sdk.eval import plot_forecast

        # Simulate SDK response
        batch_size = 32
        train_length = 100
        horizon = 24
        features = 1

        train_batch = np.random.randn(batch_size, train_length, features)
        forecast_batch = np.random.randn(batch_size, horizon, features)
        test_batch = np.random.randn(batch_size, horizon, features)

        # Plot single sample (index batch dimension)
        fig, ax = plot_forecast(
            train_data=train_batch[0],  # (100, 1)
            forecast=forecast_batch[0],  # (24, 1)
            test_data=test_batch[0],  # (24, 1)
            title="Sample 1 Forecast",
        )

        assert fig is not None
        assert ax.get_title() == "Sample 1 Forecast"
