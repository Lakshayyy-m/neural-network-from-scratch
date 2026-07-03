import numpy as np
import pytest

from loss import Loss, Loss_CategoricalCrossentropy


class TestLossBase:
    def test_calculate_returns_mean_of_sample_losses(self):
        class MockLoss(Loss):
            def forward(self, output, y):
                return np.array([1.0, 2.0, 3.0])

        loss = MockLoss()
        result = loss.calculate(None, None)
        np.testing.assert_almost_equal(result, 2.0)

    def test_calculate_single_sample(self):
        class MockLoss(Loss):
            def forward(self, output, y):
                return np.array([5.0])

        loss = MockLoss()
        result = loss.calculate(None, None)
        np.testing.assert_almost_equal(result, 5.0)

    def test_calculate_returns_scalar(self):
        class MockLoss(Loss):
            def forward(self, output, y):
                return np.array([1.0, 2.0])

        loss = MockLoss()
        result = loss.calculate(None, None)
        assert np.isscalar(result) or result.ndim == 0


class TestCategoricalCrossentropyForward:
    def setup_method(self):
        self.loss_fn = Loss_CategoricalCrossentropy()

    def test_perfect_prediction_scalar_labels(self):
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_true = np.array([0, 1, 2])
        losses = self.loss_fn.forward(y_pred, y_true)
        np.testing.assert_array_almost_equal(losses, [0.0, 0.0, 0.0], decimal=5)

    def test_wrong_prediction_high_loss(self):
        y_pred = np.array([[0.01, 0.01, 0.98]])
        y_true = np.array([0])
        losses = self.loss_fn.forward(y_pred, y_true)
        assert losses[0] > 4.0

    def test_clipping_prevents_log_zero(self):
        y_pred = np.array([[0.0, 1.0, 0.0]])
        y_true = np.array([0])
        losses = self.loss_fn.forward(y_pred, y_true)
        assert not np.any(np.isinf(losses))
        assert not np.any(np.isnan(losses))

    def test_clipping_prevents_log_one_exact(self):
        y_pred = np.array([[1.0, 0.0, 0.0]])
        y_true = np.array([0])
        losses = self.loss_fn.forward(y_pred, y_true)
        assert losses[0] >= 0

    def test_uniform_prediction(self):
        y_pred = np.array([[1 / 3, 1 / 3, 1 / 3]])
        y_true = np.array([0])
        losses = self.loss_fn.forward(y_pred, y_true)
        np.testing.assert_almost_equal(losses[0], -np.log(1 / 3), decimal=5)

    def test_batch_losses_shape(self):
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
        y_true = np.array([0, 1, 1])
        losses = self.loss_fn.forward(y_pred, y_true)
        assert losses.shape == (3,)

    def test_higher_confidence_lower_loss(self):
        y_pred_high = np.array([[0.9, 0.05, 0.05]])
        y_pred_low = np.array([[0.6, 0.2, 0.2]])
        y_true = np.array([0])
        loss_high = self.loss_fn.forward(y_pred_high, y_true)
        loss_low = self.loss_fn.forward(y_pred_low, y_true)
        assert loss_high[0] < loss_low[0]


class TestCategoricalCrossentropyCalculate:
    def test_calculate_with_scalar_labels(self):
        loss_fn = Loss_CategoricalCrossentropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
        y_true = np.array([0, 1, 1])
        loss = loss_fn.calculate(y_pred, y_true)
        assert isinstance(loss, (float, np.floating))
        assert loss > 0


class TestCategoricalCrossentropyBackward:
    def setup_method(self):
        self.loss_fn = Loss_CategoricalCrossentropy()

    def test_backward_output_shape(self):
        dvalues = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
        y_true = np.array([0, 1, 1])
        self.loss_fn.backward(dvalues, y_true)
        assert self.loss_fn.dinputs.shape == dvalues.shape

    def test_backward_with_scalar_labels(self):
        dvalues = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4]])
        y_true = np.array([0, 1])
        self.loss_fn.backward(dvalues, y_true)
        assert self.loss_fn.dinputs.shape == (2, 3)

    def test_backward_with_one_hot_labels(self):
        dvalues = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4]])
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        self.loss_fn.backward(dvalues, y_true)
        assert self.loss_fn.dinputs.shape == (2, 3)

    def test_backward_gradient_is_normalized(self):
        dvalues = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4]])
        y_true = np.array([0, 1])
        self.loss_fn.backward(dvalues, y_true)
        # gradient = (-y_true / dvalues) / samples
        expected_sample0 = np.array([-1.0 / 0.7, 0.0, 0.0]) / 2
        expected_sample1 = np.array([0.0, -1.0 / 0.5, 0.0]) / 2
        np.testing.assert_array_almost_equal(self.loss_fn.dinputs[0], expected_sample0)
        np.testing.assert_array_almost_equal(self.loss_fn.dinputs[1], expected_sample1)

    def test_backward_gradient_sign(self):
        dvalues = np.array([[0.7, 0.1, 0.2]])
        y_true = np.array([0])
        self.loss_fn.backward(dvalues, y_true)
        assert self.loss_fn.dinputs[0, 0] < 0

    def test_backward_non_target_gradient_zero(self):
        dvalues = np.array([[0.7, 0.1, 0.2]])
        y_true = np.array([0])
        self.loss_fn.backward(dvalues, y_true)
        np.testing.assert_almost_equal(self.loss_fn.dinputs[0, 1], 0.0)
        np.testing.assert_almost_equal(self.loss_fn.dinputs[0, 2], 0.0)
