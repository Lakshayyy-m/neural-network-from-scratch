import numpy as np
import pytest

from neuron import Layer_Dense


class TestLayerDenseInit:
    def test_weights_shape(self):
        layer = Layer_Dense(4, 3)
        assert layer.weights.shape == (4, 3)

    def test_biases_shape(self):
        layer = Layer_Dense(4, 3)
        assert layer.biases.shape == (1, 3)

    def test_biases_initialized_to_zero(self):
        layer = Layer_Dense(5, 2)
        np.testing.assert_array_equal(layer.biases, np.zeros((1, 2)))

    def test_weights_are_small(self):
        layer = Layer_Dense(10, 10)
        assert np.all(np.abs(layer.weights) < 1.0)

    def test_single_neuron(self):
        layer = Layer_Dense(3, 1)
        assert layer.weights.shape == (3, 1)
        assert layer.biases.shape == (1, 1)

    def test_single_input(self):
        layer = Layer_Dense(1, 5)
        assert layer.weights.shape == (1, 5)
        assert layer.biases.shape == (1, 5)


class TestLayerDenseForward:
    def test_output_shape_single_sample(self):
        layer = Layer_Dense(4, 3)
        inputs = np.random.randn(1, 4)
        layer.forward(inputs)
        assert layer.output.shape == (1, 3)

    def test_output_shape_batch(self):
        layer = Layer_Dense(4, 3)
        inputs = np.random.randn(5, 4)
        layer.forward(inputs)
        assert layer.output.shape == (5, 3)

    def test_forward_computation(self):
        layer = Layer_Dense(2, 2)
        layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.biases = np.array([[0.5, 0.5]])
        inputs = np.array([[1.0, 1.0]])
        layer.forward(inputs)
        # dot([1,1], [[1,2],[3,4]]) + [0.5,0.5] = [4, 6] + [0.5, 0.5] = [4.5, 6.5]
        np.testing.assert_array_almost_equal(layer.output, [[4.5, 6.5]])

    def test_forward_with_zero_inputs(self):
        layer = Layer_Dense(3, 2)
        layer.biases = np.array([[1.0, 2.0]])
        inputs = np.zeros((2, 3))
        layer.forward(inputs)
        np.testing.assert_array_almost_equal(layer.output, [[1.0, 2.0], [1.0, 2.0]])

    def test_forward_with_zero_weights(self):
        layer = Layer_Dense(3, 2)
        layer.weights = np.zeros((3, 2))
        layer.biases = np.zeros((1, 2))
        inputs = np.array([[1.0, 2.0, 3.0]])
        layer.forward(inputs)
        np.testing.assert_array_almost_equal(layer.output, [[0.0, 0.0]])

    def test_forward_preserves_batch_independence(self):
        layer = Layer_Dense(2, 2)
        layer.weights = np.eye(2)
        layer.biases = np.zeros((1, 2))
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.forward(inputs)
        np.testing.assert_array_almost_equal(layer.output[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(layer.output[1], [3.0, 4.0])

    def test_forward_multiple_calls_overwrite_output(self):
        layer = Layer_Dense(2, 2)
        layer.weights = np.eye(2)
        layer.biases = np.zeros((1, 2))
        inputs1 = np.array([[1.0, 2.0]])
        inputs2 = np.array([[5.0, 6.0]])
        layer.forward(inputs1)
        first_output = layer.output.copy()
        layer.forward(inputs2)
        assert not np.array_equal(first_output, layer.output)
        np.testing.assert_array_almost_equal(layer.output, [[5.0, 6.0]])
