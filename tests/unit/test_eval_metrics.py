"""Unit tests for faim_sdk.eval.metrics module.

Tests for MSE, MASE, and CRPS metrics with comprehensive coverage
of functionality, edge cases, and error handling.
"""

import numpy as np
import pytest

from faim_sdk.eval.metrics import crps_from_quantiles, mase, mse


class TestMSE:
    """Tests for Mean Squared Error (MSE) metric."""

    def test_mse_perfect_prediction(self):
        """Test MSE returns 0 for perfect predictions."""
        y_true = np.array([[[1.0], [2.0], [3.0]]])
        y_pred = np.array([[[1.0], [2.0], [3.0]]])

        result = mse(y_true, y_pred, reduction="mean")
        assert result == 0.0

    def test_mse_known_values(self):
        """Test MSE with known input/output values."""
        y_true = np.array([[[1.0], [2.0], [3.0]]])
        y_pred = np.array([[[2.0], [3.0], [4.0]]])

        # Expected: ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = 3/3 = 1.0
        result = mse(y_true, y_pred, reduction="mean")
        assert result == 1.0

    def test_mse_reduction_none(self):
        """Test MSE with reduction='none' returns per-sample metrics."""
        y_true = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
        y_pred = np.array(
            [
                [[2.0], [3.0]],  # Errors: 1, 1 -> MSE = 1.0
                [[3.0], [4.0]],  # Errors: 0, 0 -> MSE = 0.0
                [[6.0], [8.0]],  # Errors: 1, 2 -> MSE = (1+4)/2 = 2.5
            ]
        )

        result = mse(y_true, y_pred, reduction="none")

        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 2.5])

    def test_mse_multi_feature(self):
        """Test MSE with multiple features."""
        y_true = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
        y_pred = np.array([[[2.0, 2.0, 2.0]]])

        # Errors: (1-2)^2=1, (2-2)^2=0, (3-2)^2=1
        # MSE = (1 + 0 + 1) / 3 = 0.667
        result = mse(y_true, y_pred, reduction="mean")
        assert pytest.approx(result, rel=1e-6) == 2.0 / 3.0

    def test_mse_large_batch(self):
        """Test MSE with realistic batch size."""
        batch_size = 32
        horizon = 24
        features = 5

        y_true = np.random.randn(batch_size, horizon, features)
        y_pred = y_true + np.random.randn(batch_size, horizon, features) * 0.1

        result = mse(y_true, y_pred, reduction="mean")
        assert isinstance(result, float)
        assert result > 0

    def test_mse_wrong_shape_error(self):
        """Test MSE raises error for mismatched shapes."""
        y_true = np.array([[[1.0], [2.0]]])
        y_pred = np.array([[[1.0]]])  # Wrong horizon

        with pytest.raises(ValueError, match="must have the same shape"):
            mse(y_true, y_pred)

    def test_mse_wrong_dimensions_error(self):
        """Test MSE raises error for wrong number of dimensions."""
        y_true = np.array([[1.0, 2.0]])  # 2D instead of 3D
        y_pred = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="must be 3-dimensional"):
            mse(y_true, y_pred)

    def test_mse_type_error(self):
        """Test MSE raises error for non-numpy inputs."""
        y_true = [[1.0, 2.0]]  # Python list
        y_pred = np.array([[[1.0], [2.0]]])

        with pytest.raises(TypeError, match="must be numpy.ndarray"):
            mse(y_true, y_pred)

    def test_mse_empty_array_error(self):
        """Test MSE raises error for empty arrays."""
        y_true = np.array([]).reshape(1, 0, 1)
        y_pred = np.array([]).reshape(1, 0, 1)

        with pytest.raises(ValueError, match="cannot be empty"):
            mse(y_true, y_pred)

    def test_mse_invalid_reduction_error(self):
        """Test MSE raises error for invalid reduction parameter."""
        y_true = np.array([[[1.0]]])
        y_pred = np.array([[[1.0]]])

        with pytest.raises(ValueError, match="reduction must be 'mean' or 'none'"):
            mse(y_true, y_pred, reduction="invalid")


class TestMASE:
    """Tests for Mean Absolute Scaled Error (MASE) metric."""

    def test_mase_perfect_prediction(self):
        """Test MASE returns 0 for perfect predictions."""
        y_train = np.array([[[1.0], [2.0], [3.0], [4.0]]])
        y_true = np.array([[[5.0], [6.0]]])
        y_pred = np.array([[[5.0], [6.0]]])

        result = mase(y_true, y_pred, y_train, reduction="mean")
        assert result == 0.0

    def test_mase_known_values(self):
        """Test MASE with known input/output values."""
        # Training: [1, 2, 3, 4, 5] -> naive MAE = mean(|1|, |1|, |1|, |1|) = 1.0
        y_train = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

        # Test: true=[6, 7, 8], pred=[6.1, 7.1, 8.1]
        # Forecast MAE = mean(|0.1|, |0.1|, |0.1|) = 0.1
        # MASE = 0.1 / 1.0 = 0.1
        y_true = np.array([[[6.0], [7.0], [8.0]]])
        y_pred = np.array([[[6.1], [7.1], [8.1]]])

        result = mase(y_true, y_pred, y_train, reduction="mean")
        assert pytest.approx(result, rel=1e-6) == 0.1

    def test_mase_worse_than_naive(self):
        """Test MASE > 1 when forecast is worse than naive baseline."""
        # Training: constant changes of 1
        y_train = np.array([[[1.0], [2.0], [3.0], [4.0]]])

        # Bad forecast: errors of 2
        y_true = np.array([[[5.0], [6.0]]])
        y_pred = np.array([[[7.0], [8.0]]])

        # Naive MAE = 1.0, Forecast MAE = 2.0, MASE = 2.0
        result = mase(y_true, y_pred, y_train, reduction="mean")
        assert result == 2.0

    def test_mase_reduction_none(self):
        """Test MASE with reduction='none' returns per-sample metrics."""
        y_train = np.array([[[1.0], [2.0], [3.0]], [[2.0], [4.0], [6.0]], [[5.0], [5.0], [5.0]]])
        y_true = np.array([[[4.0]], [[8.0]], [[5.0]]])
        y_pred = np.array(
            [
                [[4.5]],  # Error: 0.5, Naive MAE: 1.0 -> MASE: 0.5
                [[9.0]],  # Error: 1.0, Naive MAE: 2.0 -> MASE: 0.5
                [[6.0]],  # Error: 1.0, Naive MAE: 0.0 -> MASE: inf (constant)
            ]
        )

        result = mase(y_true, y_pred, y_train, reduction="none")

        assert result.shape == (3,)
        # Third sample has constant training series, so MASE will be very high
        assert result[0] == pytest.approx(0.5, rel=1e-6)
        assert result[1] == pytest.approx(0.5, rel=1e-6)
        # Third element will be inf or very large due to constant series

    def test_mase_multi_feature(self):
        """Test MASE with multiple features."""
        y_train = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
        y_true = np.array([[[4.0, 40.0]]])
        y_pred = np.array([[[4.1, 40.5]]])

        result = mase(y_true, y_pred, y_train, reduction="mean")
        assert isinstance(result, float)
        assert result > 0

    def test_mase_constant_training_warning(self):
        """Test MASE warns about constant training series."""
        y_train = np.array([[[5.0], [5.0], [5.0]]])  # Constant
        y_true = np.array([[[5.0]]])
        y_pred = np.array([[[6.0]]])

        with pytest.warns(RuntimeWarning, match="Naive baseline MAE is zero"):
            result = mase(y_true, y_pred, y_train, reduction="mean")
            # Should return large value due to epsilon
            assert result > 0

    def test_mase_shape_mismatch_error(self):
        """Test MASE raises error for mismatched shapes."""
        y_train = np.array([[[1.0], [2.0]]])
        y_true = np.array([[[3.0]], [[4.0]]])  # Wrong batch size
        y_pred = np.array([[[3.0]]])

        with pytest.raises(ValueError, match="Batch size mismatch"):
            mase(y_true, y_pred, y_train)

    def test_mase_feature_mismatch_error(self):
        """Test MASE raises error for mismatched features."""
        y_train = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        y_true = np.array([[[5.0]]])  # Wrong features
        y_pred = np.array([[[5.0]]])

        with pytest.raises(ValueError, match="Feature count mismatch"):
            mase(y_true, y_pred, y_train)

    def test_mase_short_training_error(self):
        """Test MASE raises error for training data that's too short."""
        y_train = np.array([[[1.0]]])  # Only 1 time step
        y_true = np.array([[[2.0]]])
        y_pred = np.array([[[2.0]]])

        with pytest.raises(ValueError, match="must have at least 2 time steps"):
            mase(y_true, y_pred, y_train)

    def test_mase_type_error(self):
        """Test MASE raises error for non-numpy inputs."""
        y_train = [[1.0, 2.0]]
        y_true = np.array([[[3.0]]])
        y_pred = np.array([[[3.0]]])

        with pytest.raises(TypeError, match="must be numpy.ndarray"):
            mase(y_true, y_pred, y_train)


class TestCRPS:
    """Tests for Continuous Ranked Probability Score (CRPS) metric."""

    def test_crps_perfect_prediction(self):
        """Test CRPS returns 0 for perfect predictions."""
        y_true = np.array([[[5.0]]])
        quantile_preds = np.array([[[4.5, 5.0, 5.5]]])  # 10th, 50th, 90th
        quantile_levels = [0.1, 0.5, 0.9]

        result = crps_from_quantiles(y_true, quantile_preds, quantile_levels, reduction="mean")
        assert result == 0.0

    def test_crps_known_values(self):
        """Test CRPS with known input/output values."""
        y_true = np.array([[[5.0]]])
        quantile_preds = np.array([[[4.0, 5.0, 6.0]]])
        quantile_levels = [0.1, 0.5, 0.9]

        # Weights calculation:
        # w[0] = 0.5 - 0.1 = 0.4
        # w[1] = (0.9 - 0.1) / 2 = 0.4
        # w[2] = 0.9 - 0.5 = 0.4
        # Normalized: all 0.4 / 1.2 = 1/3

        # Errors: |5-4| = 1, |5-5| = 0, |5-6| = 1
        # CRPS = (1/3) * 1 + (1/3) * 0 + (1/3) * 1 = 2/3
        result = crps_from_quantiles(y_true, quantile_preds, quantile_levels, reduction="mean")
        assert pytest.approx(result, rel=1e-6) == 2.0 / 3.0

    def test_crps_single_quantile_equals_mae(self):
        """Test CRPS with single quantile equals MAE."""
        y_true = np.array([[[5.0], [6.0]]])
        quantile_preds = np.array([[[5.5], [6.5]]])
        quantile_levels = [0.5]

        # Should equal MAE: (|5-5.5| + |6-6.5|) / 2 = 0.5
        result = crps_from_quantiles(y_true, quantile_preds, quantile_levels, reduction="mean")
        assert pytest.approx(result, rel=1e-6) == 0.5

    def test_crps_reduction_none(self):
        """Test CRPS with reduction='none' returns per-sample metrics."""
        y_true = np.array([[[5.0]], [[6.0]]])
        quantile_preds = np.array(
            [
                [[5.0]],  # Perfect prediction
                [[7.0]],  # Error of 1
            ]
        )
        quantile_levels = [0.5]

        result = crps_from_quantiles(y_true, quantile_preds, quantile_levels, reduction="none")

        assert result.shape == (2,)
        assert result[0] == 0.0
        assert result[1] == 1.0

    def test_crps_multiple_quantiles(self):
        """Test CRPS with many quantiles."""
        y_true = np.array([[[50.0]]])
        quantile_preds = np.array([[[40.0, 45.0, 50.0, 55.0, 60.0]]])
        quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

        result = crps_from_quantiles(y_true, quantile_preds, quantile_levels, reduction="mean")

        # Median is perfect, so CRPS should be positive but not too large
        assert isinstance(result, float)
        assert result > 0
        assert result < 20  # Reasonable bound

    def test_crps_batch_processing(self):
        """Test CRPS with realistic batch size."""
        batch_size = 32
        horizon = 24
        num_quantiles = 9

        y_true = np.random.randn(batch_size, horizon, 1) * 10 + 50
        quantile_preds = np.random.randn(batch_size, horizon, num_quantiles) * 10 + 50
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        result = crps_from_quantiles(y_true, quantile_preds, quantile_levels, reduction="mean")

        assert isinstance(result, float)
        assert result > 0

    def test_crps_shape_mismatch_error(self):
        """Test CRPS raises error for mismatched shapes."""
        y_true = np.array([[[5.0]]])
        quantile_levels = [0.1, 0.5]

        with pytest.raises(ValueError, match="Batch size mismatch"):
            crps_from_quantiles(
                y_true,
                np.array([[[4.0, 5.0]], [[4.0, 5.0]]]),  # Wrong batch
                quantile_levels,
            )

    def test_crps_quantile_length_mismatch_error(self):
        """Test CRPS raises error when quantile_levels length doesn't match."""
        y_true = np.array([[[5.0]]])
        quantile_preds = np.array([[[4.0, 5.0, 6.0]]])
        quantile_levels = [0.1, 0.9]  # Only 2, but preds has 3

        with pytest.raises(ValueError, match="quantile_levels length.*must match"):
            crps_from_quantiles(y_true, quantile_preds, quantile_levels)

    def test_crps_invalid_quantile_levels_error(self):
        """Test CRPS raises error for invalid quantile levels."""
        y_true = np.array([[[5.0]]])
        quantile_preds = np.array([[[4.0, 5.0]]])
        quantile_levels = [0.1, 1.5]  # 1.5 is invalid

        with pytest.raises(ValueError, match="quantile_levels must be in"):
            crps_from_quantiles(y_true, quantile_preds, quantile_levels)

    def test_crps_unsorted_quantile_levels_error(self):
        """Test CRPS raises error for unsorted quantile levels."""
        y_true = np.array([[[5.0]]])
        quantile_preds = np.array([[[4.0, 5.0]]])
        quantile_levels = [0.9, 0.1]  # Not sorted

        with pytest.raises(ValueError, match="must be sorted in ascending order"):
            crps_from_quantiles(y_true, quantile_preds, quantile_levels)

    def test_crps_multi_feature_error(self):
        """Test CRPS raises error for multi-feature with wrong dimensions."""
        y_true = np.array([[[5.0, 6.0, 7.0]]])  # 3 features
        quantile_preds = np.array([[[4.0, 5.0]]])  # 2 quantiles
        quantile_levels = [0.1, 0.9]

        with pytest.raises(ValueError, match="features.*should be 1 or match num_quantiles"):
            crps_from_quantiles(y_true, quantile_preds, quantile_levels)

    def test_crps_type_error_quantile_levels(self):
        """Test CRPS raises error for wrong type of quantile_levels."""
        y_true = np.array([[[5.0]]])
        quantile_preds = np.array([[[4.0, 5.0]]])
        quantile_levels = np.array([0.1, 0.9])  # Should be list, not array

        with pytest.raises(TypeError, match="quantile_levels must be a list"):
            crps_from_quantiles(y_true, quantile_preds, quantile_levels)

    def test_crps_empty_array_error(self):
        """Test CRPS raises error for empty arrays."""
        y_true = np.array([]).reshape(1, 0, 1)
        quantile_preds = np.array([]).reshape(1, 0, 3)
        quantile_levels = [0.1, 0.5, 0.9]

        with pytest.raises(ValueError, match="cannot be empty"):
            crps_from_quantiles(y_true, quantile_preds, quantile_levels)
