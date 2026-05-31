import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Loss import Loss_CategoricalCrossentropy

nnfs.init()

np.random.seed(0)

# X = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # this ensures that we keeping the dimensions the same, meaning that the max is not calculated overall the batch outputs but per outputs. And then ofcourse the axis is related to which axis to calcualte the max in(here we are taking it as row)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


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

accuracy = np.mean(predictions ==y)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
