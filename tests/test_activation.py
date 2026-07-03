import numpy as np
import pytest

from activation import Activation_ReLU, Activation_Softmax


class TestActivationReLU:
    def setup_method(self):
        self.relu = Activation_ReLU()

    def test_positive_values_pass_through(self):
        inputs = np.array([[1.0, 2.0, 3.0]])
        self.relu.forward(inputs)
        np.testing.assert_array_equal(self.relu.output, [[1.0, 2.0, 3.0]])

    def test_negative_values_become_zero(self):
        inputs = np.array([[-1.0, -2.0, -3.0]])
        self.relu.forward(inputs)
        np.testing.assert_array_equal(self.relu.output, [[0.0, 0.0, 0.0]])

    def test_zero_stays_zero(self):
        inputs = np.array([[0.0, 0.0]])
        self.relu.forward(inputs)
        np.testing.assert_array_equal(self.relu.output, [[0.0, 0.0]])

    def test_mixed_values(self):
        inputs = np.array([[-1.0, 0.0, 1.0, -5.0, 3.0]])
        self.relu.forward(inputs)
        np.testing.assert_array_equal(self.relu.output, [[0.0, 0.0, 1.0, 0.0, 3.0]])

    def test_batch_processing(self):
        inputs = np.array([[-1.0, 2.0], [3.0, -4.0], [0.0, 0.5]])
        self.relu.forward(inputs)
        expected = np.array([[0.0, 2.0], [3.0, 0.0], [0.0, 0.5]])
        np.testing.assert_array_equal(self.relu.output, expected)

    def test_output_shape_preserved(self):
        inputs = np.random.randn(10, 5)
        self.relu.forward(inputs)
        assert self.relu.output.shape == (10, 5)

    def test_large_positive_values(self):
        inputs = np.array([[1e6, 1e10]])
        self.relu.forward(inputs)
        np.testing.assert_array_equal(self.relu.output, [[1e6, 1e10]])

    def test_large_negative_values(self):
        inputs = np.array([[-1e6, -1e10]])
        self.relu.forward(inputs)
        np.testing.assert_array_equal(self.relu.output, [[0.0, 0.0]])

    def test_does_not_modify_input(self):
        inputs = np.array([[-1.0, 2.0, -3.0]])
        original = inputs.copy()
        self.relu.forward(inputs)
        np.testing.assert_array_equal(inputs, original)


class TestActivationSoftmaxForward:
    def setup_method(self):
        self.softmax = Activation_Softmax()

    def test_output_sums_to_one(self):
        inputs = np.array([[1.0, 2.0, 3.0]])
        self.softmax.forward(inputs)
        np.testing.assert_almost_equal(np.sum(self.softmax.output, axis=1), [1.0])

    def test_output_all_positive(self):
        inputs = np.array([[-1.0, -2.0, -3.0]])
        self.softmax.forward(inputs)
        assert np.all(self.softmax.output > 0)

    def test_output_between_zero_and_one(self):
        inputs = np.array([[1.0, 2.0, 3.0]])
        self.softmax.forward(inputs)
        assert np.all(self.softmax.output >= 0)
        assert np.all(self.softmax.output <= 1)

    def test_highest_input_gets_highest_probability(self):
        inputs = np.array([[1.0, 5.0, 2.0]])
        self.softmax.forward(inputs)
        assert np.argmax(self.softmax.output) == 1

    def test_equal_inputs_give_equal_outputs(self):
        inputs = np.array([[1.0, 1.0, 1.0]])
        self.softmax.forward(inputs)
        np.testing.assert_array_almost_equal(
            self.softmax.output, [[1 / 3, 1 / 3, 1 / 3]]
        )

    def test_batch_sums_to_one(self):
        inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 1.0]])
        self.softmax.forward(inputs)
        row_sums = np.sum(self.softmax.output, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0])

    def test_numerical_stability_large_values(self):
        inputs = np.array([[1000.0, 1001.0, 1002.0]])
        self.softmax.forward(inputs)
        row_sum = np.sum(self.softmax.output, axis=1)
        np.testing.assert_almost_equal(row_sum, [1.0])
        assert not np.any(np.isnan(self.softmax.output))
        assert not np.any(np.isinf(self.softmax.output))

    def test_numerical_stability_negative_large_values(self):
        inputs = np.array([[-1000.0, -999.0, -998.0]])
        self.softmax.forward(inputs)
        row_sum = np.sum(self.softmax.output, axis=1)
        np.testing.assert_almost_equal(row_sum, [1.0])

    def test_output_shape(self):
        inputs = np.random.randn(8, 4)
        self.softmax.forward(inputs)
        assert self.softmax.output.shape == (8, 4)

    def test_single_class(self):
        inputs = np.array([[5.0]])
        self.softmax.forward(inputs)
        np.testing.assert_almost_equal(self.softmax.output, [[1.0]])

    def test_two_classes(self):
        inputs = np.array([[0.0, 0.0]])
        self.softmax.forward(inputs)
        np.testing.assert_array_almost_equal(self.softmax.output, [[0.5, 0.5]])
