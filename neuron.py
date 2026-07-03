import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from loss import Loss_CategoricalCrossentropy
from activation import Activation_Softmax, Activation_ReLU

nnfs.init()

np.random.seed(0)

# X = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        if not isinstance(n_inputs, (int, np.integer)) or n_inputs <= 0:
            raise ValueError(f"'n_inputs' must be a positive integer, got {n_inputs!r}")
        if not isinstance(n_neurons, (int, np.integer)) or n_neurons <= 0:
            raise ValueError(f"'n_neurons' must be a positive integer, got {n_neurons!r}")
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"'inputs' must be a numpy array, got {type(inputs).__name__}")
        if inputs.size == 0:
            raise ValueError("'inputs' must not be empty")
        if inputs.ndim < 2:
            raise ValueError(
                f"'inputs' must be at least 2D (batch of samples), got {inputs.ndim}D"
            )
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Input feature size ({inputs.shape[1]}) does not match "
                f"layer's expected input size ({self.weights.shape[0]})"
            )
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(
    samples=100, classes=3
)  # 100 data points that has 2 features each (x and y coordinates)


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(
    3, 3
)  # Again, input needs to be 3 since the previous layer output was 3.

activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)


predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
